use std::{
    cell::RefCell,
    collections::HashSet,
    ops::Add,
    rc::{Rc, Weak},
};

use crate::function::Constant;

pub type SymbolRef = Rc<RefCell<Symbol>>;
pub type SymbolWeakRef = Weak<RefCell<Symbol>>;

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Symbol(Symbol),
    Number(f32),
    Boolean(bool),
    Param,
    Add { left: Symbol, right: Symbol },
    Sub { left: Symbol, right: Symbol },
    Div { left: Symbol, right: Symbol },
    Mul { left: Symbol, right: Symbol },
    Mod { left: Symbol, right: Symbol },
    Pow { left: Symbol, right: Symbol },
    Nil,
    Not(Symbol),
    Unm(Symbol),
    Len(Symbol),
    Return(bool),
    GetTable { table: Symbol, key: Symbol },
    Closure { index: usize },
    Table(TableId),
    //Table { items: Vec<Option<Symbol>> },
    Upvalue,
    ForIndex,
    Global(Constant),
    VarArgs,
    Arg(SymbolWeakRef),
    Concat(Vec<Symbol>),
    Unknown(StackId),
    ResolvedUnknown(Vec<Symbol>), // Vector of all possible symbols
}

#[derive(Debug)]
pub enum Operation {
    SetStack(StackId, Value),
    Call {
        func: SymbolRef,
        params: Vec<SymbolRef>,
        returns: Vec<SymbolRef>,
    },
    SetGlobal(Constant, SymbolRef),
    SetCGlobal(Constant, SymbolRef),
    SetUpvalue(SymbolRef, SymbolRef),
    SetTable {
        table: SymbolRef,
        key: SymbolRef,
        value: SymbolRef,
    },
    GetVarArgs(Vec<SymbolRef>),
}

type VariableRef = Rc<Variable>;
type VariableWeakRef = Weak<Variable>;

struct Variable {
    pub label: String,
    pub stack_id: StackId,
    pub references: Vec<Rc<VariableWeakRef>>,
}

// IR Symbol
#[derive(Debug, Clone)]
pub enum Symbol {
    Stack(StackId),
    Constant(Constant),
}
/*
pub struct Symbol {
    pub value: Value,
    // Array of symbols that reference this symbol
    pub references: Vec<SymbolWeakRef>,
    pub label: String,
    // Set to true if the symbol must be evaluated where it is defined (e.g. for upvalues)
    pub force_define: bool,
}*/
/*
impl Symbol {
    pub fn new(value: Value) -> SymbolRef {
        Rc::new(RefCell::new(Symbol {
            value,
            references: Vec::new(),
            label: String::new(),
            force_define: false,
        }))
    }

    // Returns true if this symbol is a variable type
    pub fn is_var(&self) -> bool {
        matches!(self.value, Value::VarArgs | Value::Return(_, true))
    }

    // Returns true if this symbol must be defined on its own line
    pub fn must_define(&self) -> bool {
        if self.force_define || self.references.len() > 1 {
            return true;
        }
        match self.value {
            Value::Arg(_)
            | Value::Call {
                func: _,
                params: _,
                returns: _,
            } => true,
            _ => false,
        }
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

    pub fn get_varargs(count: usize) -> SymbolRef {
        let args = (0..count).map(|_| Symbol::none()).collect();
        let getva = Symbol::new(Value::GetVarArgs(args));

        if let Value::GetVarArgs(args) = &getva.borrow().value {
            for v in args {
                v.borrow_mut().value = Value::Arg(Rc::downgrade(&getva));
            }
        }
        getva
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
        } = &call.borrow().value
        {
            for r in returns {
                r.borrow_mut().value = Value::Return(Rc::downgrade(&call), false);
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

    pub fn closure(index: usize, upvalues: &[SymbolRef]) -> SymbolRef {
        // Force upvalues to be evaluated before closure
        let val = Self::new(Value::Closure {
            index,
        });
        for upval in upvalues {
            let mut upval = upval.borrow_mut();
            upval.force_define = true;
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
}*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of symbol on the stack
pub struct StackId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a table
pub struct TableId(usize);

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
    // The most recent values set on the stack
    pub stack: Vec<Option<Value>>,
    // Array of all symbols generated in this context
    pub operations: Vec<Operation>,
    pub unknowns: Vec<SymbolRef>,
    pub stack_references: HashSet<StackId>,
    pub stack_modified: HashSet<StackId>,
}

impl IrContext {
    pub fn new() -> IrContext {
        IrContext {
            stack: Vec::new(),
            operations: Vec::new(),
            unknowns: Vec::new(),
            stack_references: HashSet::new(),
            stack_modified: HashSet::new(),
        }
    }

    #[inline]
    pub fn add_referenced<T: IntoIterator<Item = Symbol>>(&mut self, ids: T) {
        self.stack_references
            .extend(ids.into_iter().filter_map(|s| match s {
                Symbol::Stack(id) => Some(id),
                _ => None,
            }));
    }

    #[inline]
    pub fn add_modified<T: IntoIterator<Item = StackId>>(&mut self, ids: T) {
        self.stack_modified.extend(ids);
    }

    // Set most recent value on a stack variable
    pub fn set_stack(&mut self, idx: StackId, val: Value) {
        if idx.0 >= self.stack.len() {
            self.stack.resize(idx.0 + 1, None);
        }
        self.add_modified([idx]);
        self.operations.push(Operation::SetStack(idx, val));
    }

    // Get most recent value set on stack
    pub fn get_stack(&mut self, idx: StackId) -> Option<Value> {
        self.stack.get(idx.0).unwrap_or(&None).map(|t| t.clone())
    }

    // Add symbol to ir history
    /*
    pub fn add_symbol(&mut self, symbol: SymbolRef) -> SymbolRef {
        self.symbols.push(symbol.clone());
        symbol
    }

    // Make add symbol
    pub fn add(&mut self, dst: StackId, left: SymbolRef, right: SymbolRef) {
        let sum = self.add_symbol(Symbol::add(left, right));
        self.set_stack(dst, sum);
    }*/

    // Make div symbol
    pub fn div(&mut self, dst: StackId, left: Symbol, right: Symbol) {
        self.add_referenced([left, right]);
        self.set_stack(dst, Value::Div { left, right });
    }

    pub fn number(&mut self, dst: StackId, n: f32) {
        self.set_stack(dst, Value::Number(n));
    }

    /*
    pub fn make_constant(&mut self, constant: Constant) -> SymbolRef {
        self.add_symbol(Symbol::new(Value::Constant(constant)))
    }*/

    // Makes a call. If param_count is None, the arguments are vararg
    // if return_count is None, the call is multiret
    pub fn call<T: IntoIterator<Item = StackId>>(
        &mut self,
        func: Symbol,
        param_base: StackId,
        param_count: Option<usize>,
        return_base: StackId,
        return_count: Option<usize>,
    ) -> SymbolRef {
        let params = match param_count {
            None => {
                let mut p = Vec::new();
                let mut current = param_base.0;
                // Add values on stack until we find a vararg
                let mut found_va = false;
                for offset in param_base.0..self.stack.len() {
                    let val = self.get_stack(StackId::from(offset));
                    p.push(Symbol::Stack(StackId::from(offset)));
                    if matches!(val, Some(Value::VarArgs | Value::Return(true))) {
                        found_va = true;
                        break;
                    }
                }
                if !found_va {
                    panic!("Failed to find vararg");
                }
                p
            }
            Some(count) => (0..count)
                .map(|p| {
                    let s = Symbol::Stack(param_base + p);
                    s
                })
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
            if return_count == -1 {
                // Set vararg flag
                if let Value::Return(_, va) = &mut returns[0].borrow_mut().value {
                    *va = true;
                }
            }
        };
        c
    }

    pub fn closure(&mut self, dst: StackId, index: usize, upvalues: &[SymbolRef]) {
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

    pub fn get_varargs(&mut self, dst: StackId, count: usize) {
        let getva = Symbol::get_varargs(count);
        self.add_symbol(getva.clone());
        if let Value::GetVarArgs(args) = &getva.borrow().value {
            for (i, arg) in args.iter().enumerate() {
                self.set_stack(StackId::from(dst.0 + i), arg.clone());
                // No need to add symbol
            }
        };
    }
}
