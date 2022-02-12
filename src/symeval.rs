use std::{collections::HashMap, cell::RefCell, rc::{Weak, Rc}};

use crate::{function::Function, ir::{StackId, IrTree, NodeFlow, ControlCode}};


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

fn generate_scope(tree: &IrTree) {
    let mut flow = NodeFlow::new(&tree);
    loop {
        let code = flow.next();
        println!("{:?}, {}", code, flow.current);
        if matches!(code.last().unwrap(), ControlCode::EndFunction) {
            break;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScopeId(usize);

struct Scope {
    id: ScopeId,
    // All nodes *directly* in this scope
    nodes: Vec<ScopeId>,
    children: Vec<ScopeId>,
    parent: ScopeId,
}

struct ScopeTree {
    scopes: Vec<Scope>,
}

impl ScopeTree {
    pub fn subscope(&mut self, root: ScopeId) -> ScopeId {
        let id = ScopeId(self.scopes.len());
        let scope = Scope {
            id,
            nodes: Vec::new(),
            children: Vec::new(),
            parent: root,
        };
        self.scopes[root.0].children.push(id);
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