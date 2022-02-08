use std::collections::HashMap;

use crate::ir::StackId;

#[derive(Debug)]
pub struct Variable {
    label: String,
}

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

pub struct SymbolicEvaluator {
    context: RootContext,
}

impl SymbolicEvaluator {
    pub fn new() -> SymbolicEvaluator {
        SymbolicEvaluator {
            context: RootContext::new(),
        }
    }
}
