use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

type SymbolRef = Rc<RefCell<Symbol>>;
type SymbolWeakRef = Weak<RefCell<Symbol>>;

#[derive(Debug)]
pub struct Call {
    func: SymbolRef,
    params: Vec<SymbolRef>,
    returns: Vec<SymbolRef>,
}

impl Call {
    pub fn new(func: SymbolRef, params: Vec<SymbolRef>, return_count: usize) -> Rc<Call> {
        let mut call = Rc::new(Call {
            func,
            params,
            returns: (0..return_count).map(|_| Symbol::none()).collect(),
        });

        for r in &call.returns {
            r.borrow_mut().value = Value::Return(Rc::downgrade(&call));
        }
        call
    }
}

#[derive(Debug)]
pub enum Value {
    None,
    Add { left: SymbolRef, right: SymbolRef },
    Return(Weak<Call>),
    Unknown(StackId),
    ResolvedUnknown(Vec<SymbolRef>), // Vector of all possible symbols
}

// IR Symbol
#[derive(Debug)]
pub struct Symbol {
    pub value: Value,
    // Array of symbols that reference this symbol
    pub references: Vec<SymbolWeakRef>,
    pub label: String,
}

impl Symbol {
    pub fn new(value: Value) -> SymbolRef {
        Rc::new(RefCell::new(Symbol {
            value,
            references: Vec::new(),
            label: String::new(),
        }))
    }

    pub fn set_label(&mut self, label: String) {
        // Return if label is the same
        if self.label == label {
            return;
        }

        // Set label of this symbol
        self.label = label.clone();

        // Check if this symbol is an unknown value,
        // and update references values if so.
        if let Value::ResolvedUnknown(up) = &self.value {
            for value in up {
                value.borrow_mut().set_label(label.clone());
            }
        }

        // Check if any unknown values reference this symbol
        for reference in &self.references {
            if let Some(reference) = reference.upgrade() {
                let mut reference = reference.borrow_mut();
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

    pub fn add_reference(&mut self, reference: &SymbolRef) {
        self.references.push(Rc::downgrade(reference));
    }

    pub fn add(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let sum = Symbol::new(Value::Add {
            left: left.clone(),
            right: right.clone(),
        });
        left.borrow_mut().add_reference(&sum);
        right.borrow_mut().add_reference(&sum);
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

    pub fn get_stack(&mut self, idx: StackId) -> SymbolRef {
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
    pub fn call(
        &mut self,
        func: SymbolRef,
        param_base: StackId,
        param_count: usize,
        return_base: StackId,
        return_count: usize,
    ) {
        let params = (0..param_count)
            .map(|p| self.get_stack(StackId::from(param_base.0 + p)))
            .collect();
        let c = Call::new(func, params, return_count);

        for p in &c.params {
            self.set_stack(StackId::from(return_base.0 + return_count), p.clone());
        }
        // todo: add call to ir
    }
}
