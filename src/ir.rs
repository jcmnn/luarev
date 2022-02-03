use std::{
    cell::RefCell,
    ops::Add,
    rc::{Rc, Weak},
};

use crate::function::Constant;

pub type SymbolRef = Rc<RefCell<Symbol>>;
pub type SymbolWeakRef = Weak<RefCell<Symbol>>;

#[derive(Debug)]
pub enum Value {
    None,
    Constant(Constant),
    Number(f32),
    Boolean(bool),
    Add {
        left: SymbolRef,
        right: SymbolRef,
    },
    Sub {
        left: SymbolRef,
        right: SymbolRef,
    },
    Div {
        left: SymbolRef,
        right: SymbolRef,
    },
    Mul {
        left: SymbolRef,
        right: SymbolRef,
    },
    Mod {
        left: SymbolRef,
        right: SymbolRef,
    },
    Pow {
        left: SymbolRef,
        right: SymbolRef,
    },
    Nil,
    Not(SymbolRef),
    Unm(SymbolRef),
    Len(SymbolRef),
    Return(SymbolWeakRef),
    Call {
        func: SymbolRef,
        params: Vec<SymbolRef>,
        returns: Vec<SymbolRef>,
    },
    GetTable {
        table: SymbolRef,
        key: SymbolRef,
    },
    SetTable {
        table: SymbolRef,
        key: SymbolRef,
        value: SymbolRef,
    },
    Closure {
        index: usize,
        upvalues: Vec<SymbolRef>,
    },
    Table {
        items: Vec<Option<SymbolRef>>,
    },
    Upvalue,
    ForIndex,
    SetUpvalue(SymbolRef, SymbolRef),
    Global(Constant),
    SetGlobal(Constant, SymbolRef),
    SetCGlobal(Constant, SymbolRef),
    VarArg,
    Concat(Vec<SymbolRef>),
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
    // Set to true if the symbol must be evaluated where it is defined (e.g. for upvalues)
    pub must_define: bool,
}

impl Symbol {
    pub fn new(value: Value) -> SymbolRef {
        Rc::new(RefCell::new(Symbol {
            value,
            references: Vec::new(),
            label: String::new(),
            must_define: false,
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

    pub fn call(func: SymbolRef, params: Vec<SymbolRef>, return_count: usize) -> SymbolRef {
        let call = Symbol::new(Value::Call {
            func: func.clone(),
            params: params.clone(),
            returns: (0..return_count).map(|_| Symbol::none()).collect(),
        });

        func.borrow_mut().add_reference(&call);

        if let Value::Call {
            func: _,
            params,
            returns,
        } = &call.borrow_mut().value
        {
            for r in returns {
                r.borrow_mut().value = Value::Return(Rc::downgrade(&call));
            }
            for param in params {
                param.borrow_mut().add_reference(&call);
            }
        }

        call
    }

    // Returns an empty symbol
    pub fn none() -> SymbolRef {
        Self::new(Value::None)
    }

    pub fn nil() -> SymbolRef {
        Self::new(Value::Nil)
    }

    pub fn add_reference(&mut self, reference: &SymbolRef) {
        self.references.push(Rc::downgrade(reference));
    }

    pub fn boolean(val: bool) -> SymbolRef {
        Self::new(Value::Boolean(val))
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

    pub fn sub(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let res = Symbol::new(Value::Sub {
            left: left.clone(),
            right: right.clone(),
        });
        left.borrow_mut().add_reference(&res);
        right.borrow_mut().add_reference(&res);
        res
    }

    pub fn div(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let sum = Symbol::new(Value::Div {
            left: left.clone(),
            right: right.clone(),
        });
        left.borrow_mut().add_reference(&sum);
        right.borrow_mut().add_reference(&sum);
        sum
    }

    pub fn mul(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let res = Symbol::new(Value::Mul {
            left: left.clone(),
            right: right.clone(),
        });
        left.borrow_mut().add_reference(&res);
        right.borrow_mut().add_reference(&res);
        res
    }

    pub fn modulus(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let res = Symbol::new(Value::Mod {
            left: left.clone(),
            right: right.clone(),
        });
        left.borrow_mut().add_reference(&res);
        right.borrow_mut().add_reference(&res);
        res
    }

    pub fn pow(left: SymbolRef, right: SymbolRef) -> SymbolRef {
        let res = Symbol::new(Value::Pow {
            left: left.clone(),
            right: right.clone(),
        });
        left.borrow_mut().add_reference(&res);
        right.borrow_mut().add_reference(&res);
        res
    }

    pub fn closure(index: usize, upvalues: Vec<SymbolRef>) -> SymbolRef {
        // Force upvalues to be evaluated before closure
        let val = Self::new(Value::Closure {
            index,
            upvalues: upvalues.clone(),
        });
        for upval in &upvalues {
            let mut upval = upval.borrow_mut();
            upval.must_define = true;
            upval.add_reference(&val);
        }
        val
    }

    pub fn concat(values: Vec<SymbolRef>) -> SymbolRef {
        let val = Self::new(Value::Concat(values.clone()));
        for v in &values {
            v.borrow_mut().add_reference(&val);
        }

        val
    }

    pub fn gettable(table: SymbolRef, key: SymbolRef) -> SymbolRef {
        let val = Self::new(Value::GetTable {
            table: table.clone(),
            key: key.clone(),
        });
        table.borrow_mut().add_reference(&val);
        key.borrow_mut().add_reference(&val);
        val
    }

    pub fn settable(table: SymbolRef, key: SymbolRef, value: SymbolRef) -> SymbolRef {
        let val = Self::new(Value::SetTable {
            table: table.clone(),
            key: key.clone(),
            value: value.clone(),
        });
        table.borrow_mut().add_reference(&val);
        key.borrow_mut().add_reference(&val);
        value.borrow_mut().add_reference(&val);
        val
    }

    pub fn set_global(key: Constant, value: SymbolRef) -> SymbolRef {
        let res = Self::new(Value::SetGlobal(key, value.clone()));
        value.borrow_mut().add_reference(&res);
        res
    }

    pub fn set_cglobal(key: Constant, value: SymbolRef) -> SymbolRef {
        let res = Self::new(Value::SetCGlobal(key, value.clone()));
        value.borrow_mut().add_reference(&res);
        res
    }

    pub fn upvalue() -> SymbolRef {
        Self::new(Value::Upvalue)
    }

    pub fn set_upvalue(upvalue: SymbolRef, value: SymbolRef) -> SymbolRef {
        let res = Self::new(Value::SetUpvalue(upvalue.clone(), value.clone()));
        upvalue.borrow_mut().add_reference(&res);
        value.borrow_mut().add_reference(&res);
        res
    }

    pub fn not(org: SymbolRef) -> SymbolRef {
        let res = Self::new(Value::Not(org.clone()));
        org.borrow_mut().add_reference(&res);
        res
    }

    pub fn unm(org: SymbolRef) -> SymbolRef {
        let res = Self::new(Value::Unm(org.clone()));
        org.borrow_mut().add_reference(&res);
        res
    }

    pub fn len(org: SymbolRef) -> SymbolRef {
        let res = Self::new(Value::Len(org.clone()));
        org.borrow_mut().add_reference(&res);
        res
    }
}

#[derive(Debug)]
// Id of symbol on the stack
pub struct StackId(usize);

impl From<u32> for StackId {
    fn from(t: u32) -> Self {
        StackId(t as usize)
    }
}

impl From<i32> for StackId {
    fn from(t: i32) -> Self {
        StackId(t as usize)
    }
}

impl From<usize> for StackId {
    fn from(t: usize) -> Self {
        StackId(t)
    }
}

impl<T: Into<StackId>> Add<T> for StackId {
    type Output = StackId;

    fn add(self, rhs: T) -> Self::Output {
        StackId(self.0 + rhs.into().0)
    }
}

// Context of IR instructions
#[derive(Debug)]
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
            self.stack.resize(idx.0 + 1, None);
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

    // Add symbol to ir history
    pub fn add_symbol(&mut self, symbol: SymbolRef) -> SymbolRef {
        self.symbols.push(symbol.clone());
        symbol
    }

    // Make add symbol
    pub fn add(&mut self, dst: StackId, left: SymbolRef, right: SymbolRef) {
        let sum = self.add_symbol(Symbol::add(left, right));
        self.set_stack(dst, sum);
    }

    // Make div symbol
    pub fn div(&mut self, dst: StackId, left: SymbolRef, right: SymbolRef) {
        let res = self.add_symbol(Symbol::div(left, right));
        self.set_stack(dst, res);
    }

    pub fn set_number(&mut self, dst: StackId, n: f32) {
        let res = self.add_symbol(Symbol::new(Value::Number(n)));
        self.set_stack(dst, res);
    }

    pub fn make_constant(&mut self, constant: Constant) -> SymbolRef {
        self.add_symbol(Symbol::new(Value::Constant(constant)))
    }

    // Make call
    pub fn call(
        &mut self,
        func: SymbolRef,
        param_base: StackId,
        param_count: i32,
        return_base: StackId,
        return_count: i32,
    ) -> SymbolRef {
        let params = match param_count {
            -1 => {
                // TODO: Check if top of stack is vararg
                [Symbol::new(Value::VarArg)].to_vec()
            }
            _ => (0..param_count as usize)
                .map(|p| self.get_stack(StackId::from(param_base.0 + p)))
                .collect(),
        };

        let c = self.add_symbol(Symbol::call(
            func,
            params,
            if return_count == -1 {
                1
            } else {
                return_count as usize
            },
        ));

        // Then add return values
        if let Value::Call {
            func: _,
            params: _,
            returns,
        } = &c.borrow().value
        {
            for (ri, p) in returns.iter().enumerate() {
                self.set_stack(StackId::from(return_base.0 + ri), p.clone());
            }
        };
        c
    }

    pub fn closure(&mut self, dst: StackId, index: usize, upvalues: Vec<SymbolRef>) {
        let val = self.add_symbol(Symbol::closure(index, upvalues));
        self.set_stack(dst, val);
    }

    pub fn concat(&mut self, dst: StackId, values: Vec<SymbolRef>) {
        let val = self.add_symbol(Symbol::concat(values));
        self.set_stack(dst, val);
    }

    pub fn gettable(&mut self, dst: StackId, table: SymbolRef, key: SymbolRef) {
        let val = self.add_symbol(Symbol::gettable(table, key));
        self.set_stack(dst, val);
    }

    pub fn set_global(&mut self, key: Constant, val: StackId) {
        let val = self.get_stack(val);
        self.add_symbol(Symbol::set_global(key, val));
    }
}
