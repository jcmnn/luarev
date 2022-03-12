use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    fs::File,
    rc::{Rc, Weak},
};

use crate::{
    function::Function,
    ir::{
        ConditionalA, ConditionalB, IrFunction, IrNode, Operation, OperationId, RegConst, StackId,
        Tail, Value, VarConst, VariableRef, VariableSolver,
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
    pub tree: &'a IrFunction<'a>,
    pub source: Vec<SourceControl>,
    scopes: ScopeTree,
    current_scope: ScopeId,
    node_scope: HashMap<usize, ScopeId>,
    solver: &'a VariableSolver,
    closures: Vec<NodeFlow<'a>>,
}

impl Display for SourceBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print(f)
    }
}

impl SourceBuilder<'_> {
    pub fn new<'a>(tree: &'a IrFunction, solver: &'a VariableSolver) -> SourceBuilder<'a> {
        let mut scopes = ScopeTree::new();
        SourceBuilder {
            tree,
            source: Vec::new(),
            current_scope: scopes.root(),
            scopes,
            node_scope: HashMap::new(),
            solver,
            closures: Vec::new(),
        }
    }

    fn write_var(&self, vref: &VariableRef, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = self.solver.get_variable(vref);
        let last_value = var.last_value.as_ref();
        if self.solver.should_label(vref) {
            write!(f, "var_{}", self.solver.references[vref.0].0)?;
        } else {
            self.write_value(last_value.unwrap(), f)?;
        }
        Ok(())
    }

    fn write_vc(&self, var: &VarConst, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match var {
            VarConst::Var(vref) => self.write_var(vref, f)?,
            VarConst::UpValue(upv) => write!(f, "upv_{}", upv.0)?, //self.write_vc(&self.tree.upvalues[upv.0], f)?,
            VarConst::Constant(cid) => write!(f, "{}", self.tree.func.constants[cid.0])?,
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
            Value::Return(op, true, id) => {
                if let Operation::Call {
                    func,
                    params,
                    returns,
                    is_multiret,
                    is_tforloop,
                } = &self.tree.nodes[id].operations[op.0]
                {
                    assert!(is_multiret);
                    assert!(!is_tforloop);
                    self.write_call(func, params, f)?;
                }
            }
            Value::Return(_, _, _) => {}
            Value::GetTable { table, key } => {
                self.write_vc(table, f)?;
                write!(f, "[")?;
                self.write_vc(key, f)?;
                write!(f, "]")?;
            }
            Value::Closure { index, upvalues } => {
                // TODO: Do stuff with upvalues
                write!(f, "function (")?;
                let closure = &self.closures[*index];
                for i in 0..closure.tree.func.num_params {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    let v = &closure.tree.nodes[&usize::MAX].variables[&StackId::from(i)];
                    self.write_var(v, f)?;
                }
                if closure.tree.func.is_vararg != 0 {
                    if closure.tree.func.num_params > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "...")?;
                }
                writeln!(f, ")")?;
                closure.source.print(f)?;
                writeln!(f, "end")?;
            }
            Value::Table(id, node) => {
                write!(f, "{{")?;
                for (i, val) in self.tree.nodes[node].tables[id.0].0.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    self.write_vc(val.as_ref().unwrap(), f)?;
                }
                write!(f, "}}")?;
            }
            Value::Upvalue(upvalue) => {
                self.write_vc(&self.tree.upvalues[upvalue.0], f)?
                //write!(f, "upvalue_{}", upvalue.0)?;
            }
            Value::ForIndex => {
                write!(f, "idx")?;
            }
            Value::Global(cid) => {
                self.tree.func.constants[cid.0].write_global(f)?;
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
                write!(f, ", ")?;
            }
            self.write_vc(p, f)?;
        }
        write!(f, ")")
    }

    fn print_node(&self, node: &IrNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "-- {}", node.id)?;
        for op in &node.operations {
            match op {
                crate::ir::Operation::SetStack(dst, val) => {
                    if self.solver.should_label(dst) {
                        self.write_var(dst, f)?;
                        write!(f, " = ")?;
                        self.write_value(val, f)?;
                        writeln!(f)?;
                    }
                }
                crate::ir::Operation::Call {
                    func,
                    params,
                    returns,
                    is_multiret,
                    is_tforloop,
                } => {
                    if *is_multiret || *is_tforloop {
                        continue;
                    }
                    if !returns.is_empty() {
                        for (i, r) in returns.iter().enumerate() {
                            if i != 0 {
                                write!(f, ", ")?;
                            }
                            self.write_vc(r, f)?;
                        }
                        write!(f, " = ")?;
                    }
                    self.write_call(func, &params, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetGlobal(cid, vc) => {
                    self.tree.func.constants[cid.0].write_global(f)?;
                    write!(f, " = ",)?;
                    self.write_vc(vc, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetCGlobal(cid, vc) => {
                    write!(f, "@")?;
                    self.tree.func.constants[cid.0].write_global(f)?;
                    write!(f, " = ",)?;
                    self.write_vc(vc, f)?;
                    writeln!(f)?;
                }
                crate::ir::Operation::SetUpvalue(upv, vc) => {
                    write!(f, "upv_{} = ", upv.0)?;
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
                            write!(f, ", ")?;
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
                    ControlCode::Continue => writeln!(f, "-- continue")?,
                    ControlCode::EndsPastLoop => todo!(),
                    ControlCode::Repeat => writeln!(f, "repeat")?,
                    ControlCode::TFor {
                        node,
                        call,
                        index,
                        state,
                    } => {
                        if let Operation::Call {
                            ref func,
                            params: _,
                            ref returns,
                            is_multiret,
                            is_tforloop,
                        } = self.tree.nodes[node].operations[call.0]
                        {
                            assert!(is_tforloop);
                            assert!(!is_multiret);
                            write!(f, "for ")?;
                            for (i, r) in returns.iter().enumerate() {
                                if i != 0 {
                                    write!(f, ", ")?;
                                }
                                self.write_vc(r, f)?;
                            }
                            write!(f, " in ")?;
                            self.write_vc(func, f)?;
                            write!(f, ", ")?;
                            self.write_var(index, f)?;
                            write!(f, ", ")?;
                            self.write_var(state, f)?;
                            writeln!(f, " do")?;
                        } else {
                            panic!("Operation not a call");
                        }
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
                    ControlCode::Return(params) => {
                        write!(f, "return ")?;
                        for (i, r) in params.iter().enumerate() {
                            if i != 0 {
                                write!(f, ", ")?;
                            }
                            self.write_vc(r, f)?;
                        }
                        writeln!(f)?;
                    }
                    ControlCode::TailCall(node, op) => {
                        write!(f, "return ")?;
                        let op = &self.tree.nodes[node].operations[op.0];
                        if let Operation::Call {
                            func,
                            is_multiret,
                            params,
                            returns,
                            is_tforloop,
                        } = op
                        {
                            assert!(!is_tforloop);
                            self.write_call(func, params, f)?;
                        }
                        writeln!(f)?;
                    }
                    ControlCode::Goto(id) => writeln!(f, "goto {id}")?,
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
                node: _,
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
        node: usize,
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
    Return(Vec<VarConst>),
    TailCall(usize, OperationId),
    If(Conditional),
    Until(Conditional),
    Goto(usize),
}

pub struct EndFinder<'a> {
    flow: &'a NodeFlow<'a>,
    cache: HashSet<usize>,
    common: HashSet<usize>,
}

impl EndFinder<'_> {
    pub fn new<'a>(flow: &'a NodeFlow<'a>) -> EndFinder<'a> {
        EndFinder {
            flow,
            cache: HashSet::new(),
            common: HashSet::new(),
        }
    }

    fn common_ends(&mut self, bases: &[usize]) {
        if bases.len() <= 1 {
            return;
        }
        // Do check with each base as the test node
        let mut to_check = VecDeque::from_iter(bases.iter().map(|i| *i));

        while let Some(test) = to_check.pop_front() {
            if self.flow.flowed.contains(&test) || !self.cache.insert(test) {
                // println!("Already checked {test}");
                continue;
            }
            // println!("Checking {test}");

            // Check if all bases end at the test node
            let count = bases
                .iter()
                .filter(|&&base| base == test || self.flow.node_ends_at(base, test))
                .count();
            // check if multiple bases ended at the test node
            if count > 1 {
                self.common.insert(test);
            }
            // check if *all* bases ended at the test node
            if count == bases.len() {
                // println!("All bases ended at {test}");
                // No need to continue searching this branch
                continue;
            }

            // Breadth-first search
            for i in &self.flow.tree.next[&test] {
                if self.common.contains(i) {
                    continue;
                }
                to_check.push_back(*i);
                // self.common_ends(bases, *i);
            }
        }
    }

    fn common_end(mut self, bases: &[usize]) -> Option<usize> {
        self.common_ends(bases);
        while self.common.len() > 1 {
            // println!("Last ends: {:?}", self.common);
            let bases: Vec<usize> = self.common.drain().collect();
            self.common_ends(&bases);
        }
        self.common.iter().next().map(|i| *i)
    }

    /*
    fn common_end(&mut self, first: usize, second: usize) -> Option<usize> {
        let mut common = HashSet::new();
        let mut cache = HashSet::new();
        //cache.insert(first);

        self.common_ends_impl(first, second, second, &mut cache, &mut common);
        //common.remove(&second);
        Vec::from_iter(common)
    }*/
}

#[derive(Debug)]
pub struct NodeFlow<'a> {
    pub source: SourceBuilder<'a>,
    flowed: HashSet<usize>,
    current: usize,
    tree: &'a IrFunction<'a>,
    flow: Vec<Flow>,
}

impl NodeFlow<'_> {
    pub fn new<'a>(tree: &'a IrFunction, solver: &'a VariableSolver) -> NodeFlow<'a> {
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
        if (ignore.contains(&start)) || !cache.insert(start) {
            return false;
        }

        self.tree.next[&start]
            .iter()
            .any(|i| end == *i || self.node_ends_at_impl(*i, end, ignore, cache, ignore_flowed))
    }

    fn node_ends_at(&self, start: usize, end: usize) -> bool {
        self.node_ends_at_impl(start, end, &self.flowed, &mut HashSet::new(), true)
    }

    pub fn common_end(&self, bases: &[usize]) -> Option<usize> {
        let ef = EndFinder::new(self);
        ef.common_end(bases)
    }

    fn check_last_flows(&mut self) -> bool {
        'outer: loop {
            let mut passed_loop = false;
            for (i, flow) in self.flow.iter().enumerate() {
                match flow {
                    Flow::While { cond, end }
                    | Flow::Repeat { cond, end }
                    | Flow::For { cond, end } => {
                        if *end == self.current {
                            /*if i != 0 {*/
                            self.source.add_control(ControlCode::Break);
                            if self.end_last_flow() {
                                return true;
                            }
                            // Break
                            /*} else {
                                if self.end_last_flow() {
                                    return true;
                                }
                                // End
                            }*/
                            continue 'outer;
                        } else if *cond == self.current {
                            if i != self.flow.len() - 1 {
                                // if passed_loop {
                                // TODO: Check if we actually break to the upper loop
                                //self.source.add_control(ControlCode::Break);
                                //} else {
                                self.source.add_control(ControlCode::Continue);
                                //}
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
                        if i != self.flow.len() - 1 {
                            println!("Code breaks from non-immediate if");
                        }
                        //self.current = *end;
                        if self.end_last_flow() {
                            return true;
                        }
                        continue 'outer;
                        // End
                    }
                    Flow::IfElse { a, b, end } if *end == self.current => {
                        if i != self.flow.len() - 1 {
                            println!("Code breaks from non-immediate if");
                        }
                        //self.current = *b;
                        if self.end_last_flow() {
                            return true;
                        }
                        continue 'outer;
                        // Else
                    }
                    _ => {}
                }
            }
            break;
        }
        false
    }

    fn nearest_loop_head(&self) -> Option<usize> {
        None
    }

    // Returns end point of flow on top of the stack
    fn last_flow_end(&self) -> Option<usize> {
        if let Some(last) = self.flow.last() {
            match last {
                Flow::While { cond, end }
                | Flow::Repeat { cond, end }
                | Flow::For { cond, end } => Some(*end),
                Flow::If { a, end } => Some(*end),
                Flow::IfElse { a, b, end } => Some(*end),
                Flow::Else { a, end } => Some(*end),
            }
        } else {
            None
        }
    }

    // Returns true if function ended
    fn end_last_flow(&mut self) -> bool {
        if let Some(last) = self.flow.pop() {
            match last {
                Flow::While { cond: _, end }
                | Flow::Repeat { cond: _, end }
                | Flow::For { cond: _, end }
                | Flow::If { a: _, end }
                | Flow::Else { a: _, end } => {
                    self.current = end;
                    self.source.add_control(ControlCode::End);
                }
                Flow::IfElse { a: _, b, end } => {
                    self.flow.push(Flow::Else { a: b, end });
                    self.current = b;
                    self.source.add_control(ControlCode::Else);
                }
            }
            if self.current == usize::MAX {
                // self.source.add_control(ControlCode::EndFunction);
                self.end_last_flow()
                //true
            } else {
                false
            }
        } else {
            self.source.add_control(ControlCode::EndFunction);
            true
        }
    }

    pub fn next(&mut self) {
        loop {
            let node = &self.tree.nodes[&self.current];
            if !self.flowed.insert(self.current) {
                println!(
                    "Writing node {} multiple times... This may generate malformed source code",
                    self.current
                );
                if !matches!(node.tail, Tail::Jmp(_)) && !matches!(node.tail, Tail::Return(_)) {
                    println!("{:?}", self.tree.next);

                    //let mut dfile = File::create("/home/jacob/luarev/test.dot").unwrap();
                    //dot::render(self.tree, &mut dfile).unwrap();

                    //panic!("Made goto :(");
                    /*self.source.add_control(ControlCode::Goto(self.current));
                    if self.end_last_flow() {
                        break;
                    } else {
                        continue;
                    }*/
                }
            }
            self.source.add_node(self.current);
            //assert!(self.flowed.insert(self.current));

            match node.tail {
                Tail::None => panic!(),
                Tail::Jmp(target) => {
                    // TODO: Check if this is a repeat-until loop or a break
                    // fall through
                    //let next_node = self.tree.next[&self.current].first().unwrap();
                    self.current = target; //*next_node;
                }
                Tail::Return(ref returns) => {
                    self.source
                        .add_control(ControlCode::Return(returns.clone()));
                    if self.end_last_flow() {
                        return;
                    }
                }
                Tail::TailCall(op) => {
                    self.source
                        .add_control(ControlCode::TailCall(self.current, op));
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
                    } else {
                        // if self.node_ends_at(target_1, target_2) {
                        let common = self.common_end(&[target_1, target_2]);
                        match common {
                            Some(x) if x == target_2 => {
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
                            }
                            Some(x) if x == target_1 => {
                                // If (reverse)
                                self.flow.push(Flow::If {
                                    a: target_2,
                                    end: target_1,
                                });
                                self.current = target_2;
                                self.source
                                    .add_control(ControlCode::If(Conditional::from_tail(
                                        &node.tail, true,
                                    )));
                            }
                            _ => {
                                // If-else ?
                                // println!("Common node: {:?}", common);
                                let end = match common {
                                    Some(end) => end,
                                    _ => {
                                        // In this case, one branch probably returns or continues a loop
                                        self.last_flow_end().unwrap_or(usize::MAX)
                                    }
                                };
                                println!("IfElse {target_1}, {target_2}, {end}");
                                self.flow.push(Flow::IfElse {
                                    a: target_1,
                                    b: target_2,
                                    end,
                                });
                                self.current = target_1;
                                self.source
                                    .add_control(ControlCode::If(Conditional::from_tail(
                                        &node.tail, false,
                                    )));
                            }
                        }
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
                    self.source.add_control(ControlCode::TFor {
                        node: self.current,
                        call,
                        index: index.clone(),
                        state: state.clone(),
                    });
                    self.current = inner;
                }
            };
            if self.check_last_flows() {
                break;
            }
        }
    }

    pub fn generate<'a>(tree: &'a IrFunction, solver: &'a VariableSolver) -> NodeFlow<'a> {
        let mut flow = NodeFlow::new(tree, solver);
        flow.next();
        for closure in &tree.closures {
            flow.source.closures.push(Self::generate(closure, solver));
        }
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
