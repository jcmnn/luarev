use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Display,
    rc::{Rc, Weak},
};

use crate::{
    function::Function,
    ir::{
        ConditionalA, ConditionalB, IrNode, IrTree, OperationId, RegConst, StackId, Tail, Value,
        VarConst, VariableRef, VariableSolver,
    },
};

#[derive(Debug, PartialEq, Eq, Hash)]
// Id of a symbol
pub struct SymbolId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a node
pub struct NodeId(pub usize);

pub struct Symbol {
    references: usize,
    // Nodes this symbol was defined in
    nodes: Vec<NodeId>,
}

#[derive(Debug)]
pub enum SourceControl {
    Control(ControlCode),
    Node(usize),
}

#[derive(Debug)]
pub struct SourceBuilder<'a> {
    pub tree: &'a IrTree,
    pub source: Vec<SourceControl>,
    scopes: ScopeTree,
    current_scope: ScopeId,
    node_scope: HashMap<usize, ScopeId>,
    solver: &'a VariableSolver,
}

impl Display for SourceBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print(f)
    }
}

impl SourceBuilder<'_> {
    pub fn new<'a>(tree: &'a IrTree, solver: &'a VariableSolver) -> SourceBuilder<'a> {
        let mut scopes = ScopeTree::new();
        SourceBuilder {
            tree,
            source: Vec::new(),
            current_scope: scopes.root(),
            scopes,
            node_scope: HashMap::new(),
            solver,
        }
    }

    fn write_var(&self, var: &VariableRef, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "var_{}", self.solver.references[var.0].0)
    }

    fn write_vc(&self, var: &VarConst, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match var {
            VarConst::Var(vref) => self.write_var(vref, f)?,
            VarConst::UpValue(upv) => write!(f, "upval_{}", upv.0)?,
            VarConst::Constant(cid) => write!(f, "const_{}", cid.0)?,
            VarConst::VarArgs => write!(f, "...")?,
            VarConst::VarCall(_) => write!(f, "...")?,
        };

        Ok(())
    }

    fn write_conditional(
        &self,
        cond: &Conditional,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match cond {
            Conditional::Eq {
                left,
                right,
                direction,
            } => {
                self.write_vc(left, f)?;
                if *direction {
                    write!(f, " == ")?;
                } else {
                    write!(f, " ~= ")?;
                }
                self.write_vc(right, f)?;
            }
            Conditional::Le {
                left,
                right,
                direction,
            } => {
                self.write_vc(left, f)?;
                if *direction {
                    write!(f, " <= ")?;
                } else {
                    write!(f, " > ")?;
                }
                self.write_vc(right, f)?;
            }
            Conditional::Lt {
                left,
                right,
                direction,
            } => {
                self.write_vc(left, f)?;
                if *direction {
                    write!(f, " < ")?;
                } else {
                    write!(f, " >= ")?;
                }
                self.write_vc(right, f)?;
            }
            Conditional::TestSet {
                value,
                target,
                direction,
            } => self.write_vc(value, f)?,
            Conditional::Test { value, direction } => {
                self.write_vc(value, f)?;
            }
        }
        Ok(())
    }

    fn write_value(&self, value: &Value, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match value {
            Value::None => todo!(),
            Value::Symbol(vc) => self.write_vc(vc, f)?,
            Value::Number(n) => write!(f, "{n}")?,
            Value::Boolean(b) => write!(f, "{b}")?,
            Value::Param => {}
            Value::Add { left, right } => {
                self.write_vc(left, f)?;
                write!(f, " + ")?;
                self.write_vc(right, f)?;
            }
            Value::Sub { left, right } => {
                self.write_vc(left, f)?;
                write!(f, " - ")?;
                self.write_vc(right, f)?;
            }
            Value::Div { left, right } => {
                self.write_vc(left, f)?;
                write!(f, " / ")?;
                self.write_vc(right, f)?;
            }
            Value::Mul { left, right } => {
                self.write_vc(left, f)?;
                write!(f, " * ")?;
                self.write_vc(right, f)?;
            }
            Value::Mod { left, right } => {
                self.write_vc(left, f)?;
                write!(f, " % ")?;
                self.write_vc(right, f)?;
            }
            Value::Pow { left, right } => {
                self.write_vc(left, f)?;
                write!(f, "^")?;
                self.write_vc(right, f)?;
            }
            Value::Nil => write!(f, "nil")?,
            Value::Not(value) => {
                write!(f, "not ")?;
                self.write_vc(value, f)?;
            }
            Value::Unm(value) => {
                write!(f, "-")?;
                self.write_vc(value, f)?;
            }
            Value::Len(value) => {
                write!(f, "#")?;
                self.write_vc(value, f)?;
            }
            Value::Return(_, _) => {}
            Value::GetTable { table, key } => {
                self.write_vc(table, f)?;
                write!(f, "[")?;
                self.write_vc(key, f)?;
                write!(f, "]")?;
            }
            Value::Closure { index, upvalues } => todo!(),
            Value::Table(id) => {
                write!(f, "{{unimplemented}}")?;
            }
            Value::Upvalue(upvalue) => {
                write!(f, "upvalue_{}", upvalue.0)?;
            }
            Value::ForIndex => {
                write!(f, "idx")?;
            }
            Value::Global(cid) => {
                write!(f, "global_cid_{}", cid.0)?;
            }
            Value::VarArgs => write!(f, "...")?,
            Value::Arg(_) => {}
            Value::Concat(vals) => {
                for (i, v) in vals.iter().enumerate() {
                    if i != 0 {
                        write!(f, " .. ")?;
                    }
                    self.write_vc(v, f)?;
                }
            }
        }
        Ok(())
    }

    fn write_call(
        &self,
        func: &VarConst,
        params: &[VarConst],
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        self.write_vc(func, f)?;
        write!(f, "(")?;
        for (i, p) in params.iter().enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            self.write_vc(p, f)?;
        }
        write!(f, ")")
    }

    fn print_node(&self, node: &IrNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for op in &node.operations {
            match op {
                crate::ir::Operation::SetStack(dst, val) => {
                    self.write_var(dst, f)?;
                    write!(f, " = ")?;
                    self.write_value(val, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::Call {
                    func,
                    params,
                    returns,
                    is_multiret,
                } => {
                    if !returns.is_empty() {
                        for (i, r) in returns.iter().enumerate() {
                            if i != 0 {
                                write!(f, ",")?;
                            }
                            self.write_vc(r, f)?;
                        }
                        write!(f, " = ")?;
                    }
                    self.write_call(func, &params, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetGlobal(cid, vc) => {
                    write!(f, "gbl_{} = ", cid.0)?;
                    self.write_vc(vc, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetCGlobal(cid, vc) => {
                    write!(f, "cgbl_{} = ", cid.0)?;
                    self.write_vc(vc, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetUpvalue(upv, vc) => {
                    write!(f, "upv_{}", upv.0)?;
                    self.write_vc(vc, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetTable { table, key, value } => {
                    self.write_vc(table, f)?;
                    write!(f, "[")?;
                    self.write_vc(key, f)?;
                    write!(f, "] = ")?;
                    self.write_vc(value, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::GetVarArgs(args) => {
                    for (i, r) in args.iter().enumerate() {
                        if i != 0 {
                            write!(f, ",")?;
                        }
                        self.write_vc(r, f)?;
                    }
                    writeln!(f, " = ...")?;
                }
                crate::ir::Operation::SetList(_, _, _) => todo!(),
            }
        }
        Ok(())
    }

    pub fn print(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for source in &self.source {
            match source {
                SourceControl::Control(code) => match code {
                    ControlCode::None => {}
                    ControlCode::EndFunction => {}
                    ControlCode::End => writeln!(f, "end")?,
                    ControlCode::Else => writeln!(f, "else")?,
                    ControlCode::Break => writeln!(f, "break")?,
                    ControlCode::Continue => writeln!(f, "continue")?,
                    ControlCode::EndsPastLoop => todo!(),
                    ControlCode::Repeat => writeln!(f, "repeat")?,
                    ControlCode::TFor { call, index, state } => {
                        writeln!(f, "for _ in _ d")?;
                    }
                    ControlCode::For {
                        step,
                        limit,
                        init,
                        idx,
                    } => {
                        write!(f, "for ")?;
                        self.write_var(idx, f)?;
                        write!(f, " = ")?;
                        self.write_vc(init, f)?;
                        write!(f, ", ")?;
                        self.write_vc(limit, f)?;
                        write!(f, ", ")?;
                        self.write_vc(step, f)?;
                        writeln!(f, " do")?;
                    }
                    ControlCode::While(cond) => {
                        write!(f, "while ")?;
                        self.write_conditional(cond, f)?;
                        writeln!(f, " do")?;
                    }
                    ControlCode::If(cond) => {
                        write!(f, "if ")?;
                        self.write_conditional(cond, f)?;
                        writeln!(f, " then")?;
                    }
                    ControlCode::Until(_) => todo!(),
                },
                SourceControl::Node(node) => {
                    self.print_node(&self.tree.nodes[node], f)?;
                }
            }
        }

        Ok(())
    }

    fn add_control(&mut self, control: ControlCode) {
        match control {
            ControlCode::EndFunction => {
                assert!(self.current_scope.is_root());
            }
            ControlCode::End | ControlCode::Until(_) => {
                // Go up a scope
                self.current_scope = self.scopes.parent(self.current_scope);
            }
            ControlCode::Else => {
                // Make a new subscope of the parent
                self.current_scope = self.scopes.subscope(self.scopes.parent(self.current_scope));
            }
            ControlCode::EndsPastLoop => todo!(),
            ControlCode::If(_)
            | ControlCode::While(_)
            | ControlCode::For {
                step: _,
                limit: _,
                init: _,
                idx: _,
            }
            | ControlCode::TFor {
                call: _,
                index: _,
                state: _,
            }
            | ControlCode::Repeat => {
                // Make a new subscope
                self.current_scope = self.scopes.subscope(self.current_scope);
            }
            _ => {}
        };
        self.source.push(SourceControl::Control(control));
    }

    fn add_node(&mut self, node: usize) {
        // Add node to current scope
        self.scopes.add_node(self.current_scope, node);
        self.node_scope.insert(node, self.current_scope);
        self.source.push(SourceControl::Node(node));
    }

    fn finish(&mut self) {
        for control in &self.source {
            match control {
                SourceControl::Control(code) => {}
                SourceControl::Node(node) => {}
            }
        }
    }
}

#[derive(Debug)]
enum Flow {
    While { cond: usize, end: usize },
    Repeat { cond: usize, end: usize },
    If { a: usize, end: usize },
    IfElse { a: usize, b: usize, end: usize },
    Else { a: usize, end: usize },
    For { cond: usize, end: usize },
}

#[derive(Debug)]
pub enum Conditional {
    Eq {
        left: VarConst,
        right: VarConst,
        direction: bool,
    },
    Le {
        left: VarConst,
        right: VarConst,
        direction: bool,
    },
    Lt {
        left: VarConst,
        right: VarConst,
        direction: bool,
    },
    TestSet {
        value: VarConst,
        target: VariableRef,
        direction: bool,
    },
    Test {
        value: VarConst,
        direction: bool,
    },
}

impl Conditional {
    pub fn from_tail(tail: &Tail, reverse_direction: bool) -> Conditional {
        match tail {
            Tail::Eq(ConditionalB {
                left,
                right,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Eq {
                left: left.clone(),
                right: right.clone(),
                direction: direction ^ reverse_direction,
            },
            Tail::Le(ConditionalB {
                left,
                right,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Le {
                left: left.clone(),
                right: right.clone(),
                direction: direction ^ reverse_direction,
            },
            Tail::Lt(ConditionalB {
                left,
                right,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Lt {
                left: left.clone(),
                right: right.clone(),
                direction: direction ^ reverse_direction,
            },
            Tail::TestSet(
                ConditionalA {
                    value,
                    direction,
                    target_1: _,
                    target_2: _,
                },
                target,
            ) => Self::TestSet {
                value: value.clone(),
                direction: direction ^ reverse_direction,
                target: target.clone(),
            },
            Tail::Test(ConditionalA {
                value,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Test {
                value: value.clone(),
                direction: direction ^ reverse_direction,
            },
            _ => panic!("Tail is not a conditional"),
        }
    }
}

#[derive(Debug)]
pub enum ControlCode {
    None,
    EndFunction,
    End,
    Else,
    Break,
    Continue,
    EndsPastLoop,
    Repeat,
    TFor {
        call: OperationId,
        index: VariableRef,
        state: VariableRef,
    },
    For {
        step: VarConst,
        limit: VarConst,
        init: VarConst,
        idx: VariableRef,
    },
    While(Conditional),
    If(Conditional),
    Until(Conditional),
}

#[derive(Debug)]
pub struct NodeFlow<'a> {
    pub source: SourceBuilder<'a>,
    flowed: HashSet<usize>,
    current: usize,
    tree: &'a IrTree,
    flow: Vec<Flow>,
}

impl NodeFlow<'_> {
    pub fn new<'a>(tree: &'a IrTree, solver: &'a VariableSolver) -> NodeFlow<'a> {
        NodeFlow {
            source: SourceBuilder::new(tree, solver),
            flowed: HashSet::new(),
            current: 0,
            tree,
            flow: Vec::new(),
        }
    }

    fn node_ends_at_impl(
        &self,
        start: usize,
        end: usize,
        ignore: &HashSet<usize>,
        cache: &mut HashSet<usize>,
        ignore_flowed: bool,
    ) -> bool {
        if start == end {
            return true;
        }
        if (ignore.contains(&start)) || !cache.insert(start) {
            return false;
        }

        self.tree.next[&start]
            .iter()
            .any(|i| self.node_ends_at_impl(*i, end, ignore, cache, ignore_flowed))
    }

    fn node_ends_at(&self, start: usize, end: usize) -> bool {
        self.node_ends_at_impl(start, end, &self.flowed, &mut HashSet::new(), true)
    }

    fn common_ends_impl(
        &self,
        first: usize,
        second: usize,
        cache: &mut HashSet<usize>,
        common: &mut HashSet<usize>,
    ) {
        if !cache.insert(second) {
            return;
        }

        if self.node_ends_at_impl(first, second, common, &mut HashSet::new(), false) {
            common.insert(second);
        }

        for i in &self.tree.next[&second] {
            if common.contains(i) {
                continue;
            }
            self.common_ends_impl(first, *i, cache, common);
        }
    }

    fn common_ends(&self, first: usize, second: usize) -> Vec<usize> {
        let mut common = HashSet::new();
        let mut cache = HashSet::new();
        cache.insert(first);
        self.common_ends_impl(first, second, &mut cache, &mut common);
        common.remove(&second);
        Vec::from_iter(common)
    }

    fn check_last_flows(&mut self) -> bool {
        'outer: loop {
            let mut passed_loop = false;
            for (i, flow) in self.flow.iter().rev().enumerate() {
                match flow {
                    Flow::While { cond, end }
                    | Flow::Repeat { cond, end }
                    | Flow::For { cond, end } => {
                        if *end == self.current {
                            if i != 0 {
                                self.source.add_control(ControlCode::Break);
                                if self.end_last_flow() {
                                    return true;
                                }
                                // Break
                            } else {
                                if self.end_last_flow() {
                                    return true;
                                }
                                // End
                            }
                            continue 'outer;
                        } else if *cond == self.current {
                            if i != 0 {
                                if passed_loop {
                                    // TODO: Check if we actually break to the upper loop
                                    self.source.add_control(ControlCode::Break);
                                } else {
                                    self.source.add_control(ControlCode::Continue);
                                }
                            }
                            if self.end_last_flow() {
                                return true;
                            }
                            continue 'outer;
                        } else {
                            passed_loop = true;
                            // This is an error
                            // panic!("Flow goes outside loop");
                            // Some((i, ControlCode::EndsPastLoop))
                        }
                    }
                    Flow::If { a, end } | Flow::Else { a, end } if *end == self.current => {
                        if i != 0 {
                            panic!("Code breaks from non-immediate if");
                        }
                        self.current = *end;
                        if self.end_last_flow() {
                            return true;
                        }
                        continue 'outer;
                        // End
                    }
                    Flow::IfElse { a, b, end } if *end == self.current => {
                        if i != 0 {
                            panic!("Code breaks from non-immediate if");
                        }
                        self.current = *b;
                        if self.end_last_flow() {
                            return true;
                        }
                        continue 'outer;
                        // Else
                    }
                    _ => {}
                }
            }
            break 'outer;
        }
        false
    }

    // Returns true if function ended
    fn end_last_flow(&mut self) -> bool {
        if let Some(last) = self.flow.pop() {
            match last {
                Flow::While { cond, end }
                | Flow::Repeat { cond, end }
                | Flow::For { cond, end } => {
                    self.current = end;
                    self.source.add_control(ControlCode::End);
                }
                Flow::If { a, end } => {
                    self.current = end;
                    self.source.add_control(ControlCode::End);
                }
                Flow::IfElse { a, b, end } => {
                    self.flow.push(Flow::Else { a: b, end });
                    self.current = b;
                    self.source.add_control(ControlCode::Else);
                }
                Flow::Else { a, end } => {
                    self.current = end;
                    self.source.add_control(ControlCode::End);
                }
            }
            false
        } else {
            self.source.add_control(ControlCode::EndFunction);
            true
        }
    }

    pub fn next(&mut self) {
        loop {
            self.source.add_node(self.current);
            let node = &self.tree.nodes[&self.current];
            assert!(self.flowed.insert(self.current));

            match node.tail {
                Tail::None => {
                    // TODO: Check if this is a repeat-until loop or a break
                    // fall through
                    let next_node = self.tree.next[&self.current].first().unwrap();
                    self.current = *next_node;
                }
                Tail::Return(_) | Tail::TailCall(_) => {
                    if self.end_last_flow() {
                        return;
                    }
                }
                Tail::Eq(ConditionalB {
                    left: _,
                    right: _,
                    direction: _,
                    target_1,
                    target_2,
                })
                | Tail::Le(ConditionalB {
                    left: _,
                    right: _,
                    direction: _,
                    target_1,
                    target_2,
                })
                | Tail::Lt(ConditionalB {
                    left: _,
                    right: _,
                    direction: _,
                    target_1,
                    target_2,
                })
                | Tail::TestSet(
                    ConditionalA {
                        value: _,
                        direction: _,
                        target_1,
                        target_2,
                    },
                    _,
                )
                | Tail::Test(ConditionalA {
                    value: _,
                    direction: _,
                    target_1,
                    target_2,
                }) => {
                    if self.node_ends_at(target_1, self.current) {
                        // Loop
                        self.flow.push(Flow::While {
                            cond: self.current,
                            end: target_2,
                        });
                        self.current = target_1;
                        self.source
                            .add_control(ControlCode::While(Conditional::from_tail(
                                &node.tail, false,
                            )));
                    } else if self.node_ends_at(target_2, self.current) {
                        // Loop (reverse)
                        self.flow.push(Flow::While {
                            cond: self.current,
                            end: target_1,
                        });
                        self.current = target_2;
                        self.source
                            .add_control(ControlCode::While(Conditional::from_tail(
                                &node.tail, true,
                            )));
                    } else if self.node_ends_at(target_1, target_2) {
                        // If
                        self.flow.push(Flow::If {
                            a: target_1,
                            end: target_2,
                        });
                        self.current = target_1;
                        self.source
                            .add_control(ControlCode::If(Conditional::from_tail(
                                &node.tail, false,
                            )));
                    } else if self.node_ends_at(target_2, target_1) {
                        // If (reverse)
                        self.flow.push(Flow::If {
                            a: target_2,
                            end: target_1,
                        });
                        self.current = target_2;
                        self.source
                            .add_control(ControlCode::If(Conditional::from_tail(&node.tail, true)));
                    } else {
                        // If-else ?
                        let common = self.common_ends(target_1, target_2);
                        println!("Common nodes: {:?}", common);
                        if common.len() != 1 {
                            todo!();
                        }
                        self.flow.push(Flow::IfElse {
                            a: target_1,
                            b: target_2,
                            end: *common.first().unwrap(),
                        });
                        self.current = target_1;
                        self.source
                            .add_control(ControlCode::If(Conditional::from_tail(
                                &node.tail, false,
                            )));
                    }
                }
                Tail::ForLoop {
                    ref init,
                    ref limit,
                    ref step,
                    ref idx,
                    inner,
                    end,
                } => {
                    self.flow.push(Flow::For {
                        cond: self.current,
                        end,
                    });
                    self.current = inner;
                    self.source.add_control(ControlCode::For {
                        step: step.clone(),
                        limit: limit.clone(),
                        init: init.clone(),
                        idx: idx.clone(),
                    });
                }
                Tail::TForLoop {
                    call,
                    ref index,
                    ref state,
                    inner,
                    end,
                } => {
                    self.flow.push(Flow::For {
                        cond: self.current,
                        end,
                    });
                    self.current = inner;
                    self.source.add_control(ControlCode::TFor {
                        call,
                        index: index.clone(),
                        state: state.clone(),
                    });
                }
            };
            if self.check_last_flows() {
                break;
            }
        }
    }

    pub fn generate<'a>(tree: &'a IrTree, solver: &'a VariableSolver) -> NodeFlow<'a> {
        let mut flow = NodeFlow::new(tree, solver);
        flow.next();
        flow
    }
}

#[derive(Debug, Clone, Copy)]
struct ScopeId(usize);

impl ScopeId {
    pub fn is_root(&self) -> bool {
        self.0 == 0
    }
}

#[derive(Debug)]
struct Scope {
    id: ScopeId,
    // All nodes *directly* in this scope
    nodes: Vec<usize>,
    children: Vec<ScopeId>,
    parent: ScopeId,
}

#[derive(Debug)]
struct ScopeTree {
    scopes: Vec<Scope>,
}

impl ScopeTree {
    pub fn new() -> ScopeTree {
        ScopeTree {
            scopes: Vec::from_iter([Scope {
                id: ScopeId(0),
                nodes: Vec::new(),
                children: Vec::new(),
                parent: ScopeId(0),
            }]),
        }
    }

    // Returns the parent scope
    pub fn parent(&self, scope: ScopeId) -> ScopeId {
        self.scopes[scope.0].parent
    }

    // Returns the root scope
    pub fn root(&mut self) -> ScopeId {
        ScopeId(0)
    }

    // Add a node to a scope
    pub fn add_node(&mut self, scope: ScopeId, node: usize) {
        self.scopes[scope.0].nodes.push(node);
    }

    // Creates a new subscope
    pub fn subscope(&mut self, parent: ScopeId) -> ScopeId {
        let id = ScopeId(self.scopes.len());
        let scope = Scope {
            id,
            nodes: Vec::new(),
            children: Vec::new(),
            parent,
        };
        self.scopes[parent.0].children.push(id);
        self.scopes.push(scope);
        id
    }
}

pub struct NodeBuilder<'a> {
    id: NodeId,
    function: &'a FunctionBuilder<'a>,
    stack: Vec<Option<SymbolId>>,
}

pub struct NodeContext {
    id: NodeId,
    prev: Vec<NodeId>,
    stack: Vec<Option<SymbolId>>,
}

pub struct FunctionBuilder<'a> {
    function: &'a Function,
    evaluator: &'a mut SymbolicEvaluator,
    nodes: HashMap<NodeId, NodeContext>,
}

pub struct SymbolicEvaluator {
    symbols: Vec<Symbol>,
    buffer: Vec<u8>,
}

impl FunctionBuilder<'_> {
    pub fn new<'a>(
        function: &'a Function,
        evaluator: &'a mut SymbolicEvaluator,
    ) -> FunctionBuilder<'a> {
        FunctionBuilder {
            function,
            evaluator,
            nodes: HashMap::new(),
        }
    }

    pub fn node(&mut self, id: NodeId) -> NodeBuilder {
        NodeBuilder {
            id,
            function: self,
            stack: Vec::new(),
        }
    }
}

impl SymbolicEvaluator {
    pub fn new() -> SymbolicEvaluator {
        SymbolicEvaluator {
            symbols: Vec::new(),
            buffer: Vec::new(),
        }
    }

    pub fn start_function(&mut self, param_count: usize) {}

    //pub fn ref_symbol(&self)
}
