use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::{Rc, Weak},
};

use crate::{
    function::Function,
    ir::{ConditionalA, ConditionalB, IrTree, RegConst, StackId, Tail, IrNode},
};

#[derive(Debug)]
pub struct RootContext {
    variables: Vec<Variable>,
}

impl RootContext {
    pub fn new() -> RootContext {
        RootContext {
            variables: Vec::new(),
        }
    }
}

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

pub struct NodeDecompiler<'a> {
    node: &'a IrNode,
}

#[derive(Debug)]
pub enum SourceControl {
    Control(ControlCode),
    Node(usize),
}

#[derive(Debug)]
struct SourceBuilder<'a> {
    tree: &'a IrTree,
    source: Vec<SourceControl>,
    scopes: ScopeTree,
    current_scope: ScopeId,
    node_scope: HashMap<usize, ScopeId>,
}

impl SourceBuilder<'_> {
    pub fn new(tree: &IrTree) -> SourceBuilder {
        let mut scopes = ScopeTree::new();
        SourceBuilder {
            tree,
            source: Vec::new(),
            current_scope: scopes.root(),
            scopes,
            node_scope: HashMap::new(),
        }
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
            ControlCode::If(_) | ControlCode::While(_) | ControlCode::For | ControlCode::Repeat => {
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
                SourceControl::Control(code) => {

                },
                SourceControl::Node(node) => {

                },
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
        left: RegConst,
        right: RegConst,
        direction: bool,
    },
    Le {
        left: RegConst,
        right: RegConst,
        direction: bool,
    },
    Lt {
        left: RegConst,
        right: RegConst,
        direction: bool,
    },
    TestSet {
        value: RegConst,
        target: StackId,
        direction: bool,
    },
    Test {
        value: RegConst,
        direction: bool,
    },
}

impl Conditional {
    pub fn from_tail(tail: &Tail, reverse_direction: bool) -> Conditional {
        match *tail {
            Tail::Eq(ConditionalB {
                left,
                right,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Eq {
                left,
                right,
                direction: direction ^ reverse_direction,
            },
            Tail::Le(ConditionalB {
                left,
                right,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Le {
                left,
                right,
                direction: direction ^ reverse_direction,
            },
            Tail::Lt(ConditionalB {
                left,
                right,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Lt {
                left,
                right,
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
                value,
                direction: direction ^ reverse_direction,
                target,
            },
            Tail::Test(ConditionalA {
                value,
                direction,
                target_1: _,
                target_2: _,
            }) => Self::Test {
                value,
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
    For,
    While(Conditional),
    If(Conditional),
    Until(Conditional),
}

#[derive(Debug)]
pub struct NodeFlow<'a> {
    source: SourceBuilder<'a>,
    flowed: HashSet<usize>,
    pub current: usize,
    tree: &'a IrTree,
    flow: Vec<Flow>,
}

impl NodeFlow<'_> {
    pub fn new(tree: &IrTree) -> NodeFlow {
        NodeFlow {
            source: SourceBuilder::new(tree),
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
                Tail::TForLoop {
                    call: _,
                    index: _,
                    state: _,
                    inner,
                    end,
                }
                | Tail::ForLoop {
                    init: _,
                    limit: _,
                    step: _,
                    idx: _,
                    inner,
                    end,
                } => {
                    self.flow.push(Flow::For {
                        cond: self.current,
                        end,
                    });
                    self.current = inner;
                    self.source.add_control(ControlCode::For);
                }
            };
            if self.check_last_flows() {
                break;
            }
        }
    }
}

pub fn generate_scope(tree: &IrTree) {
    let mut flow = NodeFlow::new(&tree);
    flow.next();
    println!("Source: {:#?}", flow.source);
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

#[derive(Debug)]
pub struct Variable {
    register: StackId,
    label: String,
    is_static: bool,
    references: Vec<Weak<VariableRef>>,
}

#[derive(Debug)]
pub struct VariableRef {
    variable: RefCell<Weak<Variable>>,
    node: usize,
}

pub struct VariableSolver {
    variables: Vec<Rc<Variable>>,
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
    context: RootContext,
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
            context: RootContext::new(),
            symbols: Vec::new(),
            buffer: Vec::new(),
        }
    }

    pub fn start_function(&mut self, param_count: usize) {}

    //pub fn ref_symbol(&self)
}
