use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

struct Call {}

type SymbolRef = Rc<Symbol>;
type SymbolWeakRef = Weak<Symbol>;

pub enum Value {
    None,
    Add { left: SymbolRef, right: SymbolRef },
    Unknown,
    ResolvedUnknown(Vec<SymbolRef>), // Vector of all possible symbols
}

// IR Symbol
pub struct Symbol {
    pub value: Value,
    // Array of symbols that reference this symbol
    pub references: RefCell<Vec<SymbolWeakRef>>,
    pub label: RefCell<String>,
}

impl Symbol {
    pub fn new(value: Value) -> SymbolRef {
        Rc::new(Symbol {
            value,
            references: RefCell::new(Vec::new()),
            label: RefCell::new(String::new()),
        })
    }

    pub fn none() -> SymbolRef {
        Self::new(Value::None)
    }

    pub fn add_reference(&self, reference: &SymbolRef) {
        self.references.borrow_mut().push(Rc::downgrade(reference));
    }

    pub fn add(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let sum = Symbol::new(Value::Add {
            left: left.clone(),
            right: right.clone(),
        });
        left.add_reference(&sum);
        right.add_reference(&sum);
        sum
    }
}

// Id of symbol on the stack
pub struct StackId(usize);

// Context of IR instructions
pub struct IrContext {
    // Symbols on the stack
    stack: Vec<Option<SymbolRef>>,
    // Array of all symbols generated in this context
    symbols: Vec<SymbolRef>,
}

impl IrContext {
    pub fn new() -> IrContext {
        IrContext {
            stack: Vec::new(),
            symbols: Vec::new(),
        }
    }

    pub fn set_stack(&mut self, idx: StackId, val: SymbolRef) {
        if idx.0 >= self.stack.len() {
            self.stack.resize(idx.0, None);
        }
        self.stack[idx.0] = Some(val);
    }

    pub fn get_stack(&mut self, idx: StackId, val: SymbolRef) {

    }

    pub fn add(&mut self, dst: StackId, left: SymbolRef, right: SymbolRef) {
        let sum = Symbol::add(left, right);
        self.set_stack(dst, sum.clone());
        self.symbols.push(sum);
    }
}
