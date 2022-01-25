use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};


type SymbolRef = Rc<Symbol>;
type SymbolWeakRef = Weak<Symbol>;

#[derive(Debug)]
pub struct Call {
    func: SymbolRef,
    params: Vec<SymbolRef>,
    returns: Vec<SymbolRef>,
}

impl Call {
    
}

#[derive(Debug)]
pub enum Value {
    None,
    Add { left: SymbolRef, right: SymbolRef },
    Unknown(StackId),
    ResolvedUnknown(Vec<SymbolRef>), // Vector of all possible symbols
}

// IR Symbol
#[derive(Debug)]
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

    pub fn set_label(&self, label: String) {
        // Return if label is the same
        if *self.label.borrow() == label {
            return;
        }

        // Set label of this symbol
        self.label.replace(label.clone());

        // Check if this symbol is an unknown value,
        // and update references values if so.
        if let Value::ResolvedUnknown(up) = &self.value {
            for value in up {
                value.set_label(label.clone());
            }
        }

        // Check if any unknown values reference this symbol
        for reference in self.references.borrow().iter() {
            if let Some(reference) = reference.upgrade() {
                if matches!(reference.value, Value::ResolvedUnknown(_)) {
                    reference.set_label(label.clone());
                }
            }
        }
    }

    // Returns an empty symbol
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

#[derive(Debug)]
// Id of symbol on the stack
pub struct StackId(usize);

impl<T: Into<usize>> From<T> for StackId {
    fn from(t: T) -> Self {
        StackId(t.into())
    }
}

// Context of IR instructions
pub struct IrContext {
    // Symbols on the stack
    stack: Vec<Option<SymbolRef>>,
    // Array of all symbols generated in this context
    symbols: Vec<SymbolRef>,
    unknowns: Vec<SymbolRef>,
}

impl IrContext {
    pub fn new() -> IrContext {
        IrContext {
            stack: Vec::new(),
            symbols: Vec::new(),
            unknowns: Vec::new(),
        }
    }

    pub fn set_stack(&mut self, idx: StackId, val: SymbolRef) {
        if idx.0 >= self.stack.len() {
            self.stack.resize(idx.0, None);
        }
        self.stack[idx.0] = Some(val);
    }

    pub fn get_stack(&mut self, idx: StackId) -> Rc<Symbol> {
        if idx.0 >= self.stack.len() {
            self.stack.resize(idx.0, None);
        }

        let val = &mut self.stack[idx.0];

        match val {
            Some(x) => x.clone(),
            None => {
                let v = Symbol::new(Value::Unknown(idx));
                self.unknowns.push(v.clone());
                *val = Some(v.clone());
                v
            }
        }
    }

    // Make add symbol
    pub fn add(&mut self, dst: StackId, left: SymbolRef, right: SymbolRef) {
        let sum = Symbol::add(left, right);
        self.set_stack(dst, sum.clone());
        self.symbols.push(sum);
    }

    // Make call
    pub fn call(&mut self, func: SymbolRef, params: Vec<SymbolRef>, return_base: StackId, return_count: usize)
}
