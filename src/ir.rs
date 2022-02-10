use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    io::Cursor,
    io::Write,
    ops::Add,
    rc::{Rc, Weak},
};

use serde::de::value;

use crate::function::{Constant, LvmInstruction};

pub type SymbolRef = Rc<RefCell<RegConst>>;
pub type SymbolWeakRef = Weak<RefCell<RegConst>>;

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Symbol(RegConst),
    Number(f32),
    Boolean(bool),
    Param,
    Add {
        left: RegConst,
        right: RegConst,
    },
    Sub {
        left: RegConst,
        right: RegConst,
    },
    Div {
        left: RegConst,
        right: RegConst,
    },
    Mul {
        left: RegConst,
        right: RegConst,
    },
    Mod {
        left: RegConst,
        right: RegConst,
    },
    Pow {
        left: RegConst,
        right: RegConst,
    },
    Nil,
    Not(RegConst),
    Unm(RegConst),
    Len(RegConst),
    Return(OperationId, bool),
    GetTable {
        table: RegConst,
        key: RegConst,
    },
    Closure {
        index: usize,
        upvalues: Vec<RegConst>,
    },
    Table(TableId),
    //Table { items: Vec<Option<Symbol>> },
    Upvalue(UpvalueId),
    ForIndex,
    Global(ConstantId),
    VarArgs,
    Arg(OperationId),
    Concat(Vec<RegConst>),
    Unknown(StackId),
    ResolvedUnknown(Vec<RegConst>), // Vector of all possible symbols
}

impl Value {
    pub fn is_var(&self) -> bool {
        matches!(*self, Value::VarArgs | Value::Return(_, true))
    }
}

#[derive(Debug)]
pub enum Operation {
    SetStack(StackId, Value),
    Call {
        func: RegConst,
        params: Vec<RegConst>,
        returns: Vec<RegConst>,
        is_multiret: bool,
    },
    SetGlobal(ConstantId, RegConst),
    SetCGlobal(ConstantId, RegConst),
    SetUpvalue(UpvalueId, RegConst),
    SetTable {
        table: RegConst,
        key: RegConst,
        value: RegConst,
    },
    GetVarArgs(Vec<RegConst>),
    SetList(StackId, usize, Vec<RegConst>),
}

#[derive(Debug)]
pub struct Table(Vec<Option<RegConst>>);

// IR Symbol
#[derive(Debug, Clone, Copy)]
pub enum RegConst {
    Stack(StackId),
    Constant(ConstantId),
    VarArgs,
    VarCall(OperationId),
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
pub struct StackId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a table
pub struct TableId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of an operation
pub struct OperationId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a constant
pub struct ConstantId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a variable
pub struct VariableId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of an upvalue
pub struct UpvalueId(usize);

impl From<usize> for UpvalueId {
    fn from(id: usize) -> Self {
        UpvalueId(id)
    }
}

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

#[derive(Debug)]
pub struct ConditionalA {
    pub value: RegConst,
    pub direction: bool,
    pub target_1: usize,
    pub target_2: usize,
}

#[derive(Debug)]
pub struct ConditionalB {
    pub left: RegConst,
    pub right: RegConst,
    pub direction: bool,
    pub target_1: usize,
    pub target_2: usize,
}

#[derive(Debug)]
pub enum Tail {
    None,
    Return(Vec<RegConst>),
    TailCall(OperationId),
    Eq(ConditionalB),
    Le(ConditionalB),
    Lt(ConditionalB),
    TestSet(ConditionalA, StackId),
    Test(ConditionalA),
    TForLoop {
        call: OperationId,
        index: RegConst,
        state: RegConst,
        inner: usize,
        end: usize,
    },
    ForLoop {
        init: RegConst,
        limit: RegConst,
        step: RegConst,
        idx: StackId,
        inner: usize,
        end: usize,
    },
}

// Context of IR instructions
#[derive(Debug)]
pub struct IrNode {
    // The most recent operations that operated on the stack
    pub stack: Vec<Option<Value>>,
    // Array of all symbols generated in this context
    pub operations: Vec<Operation>,
    pub tables: Vec<Table>,
    pub stack_references: HashSet<StackId>,
    pub stack_modified: HashSet<StackId>,
    // Map of register to variable
    pub variables: HashMap<StackId, VariableId>,
    pub references: HashMap<StackId, VariableId>,
    pub tail: Tail,
}

impl IrNode {
    pub fn new() -> IrNode {
        IrNode {
            stack: Vec::new(),
            operations: Vec::new(),
            tables: Vec::new(),
            stack_references: HashSet::new(),
            stack_modified: HashSet::new(),
            variables: HashMap::new(),
            references: HashMap::new(),
            tail: Tail::None,
        }
    }

    #[inline]
    pub fn add_referenced<'a, T: 'a + IntoIterator<Item = &'a RegConst>>(&mut self, ids: T) {
        self.stack_references
            .extend(ids.into_iter().filter_map(|s| match s {
                RegConst::Stack(id) if !self.stack_modified.contains(id) => Some(id),
                _ => None,
            }));
    }

    #[inline]
    fn add_modified<T: IntoIterator<Item = StackId>>(&mut self, ids: T) {
        self.stack_modified.extend(ids);
    }

    // Get last stack symbols from base to the first vararg
    fn base_to_vararg(&self, base: StackId) -> Vec<RegConst> {
        let mut symbols = Vec::new();
        for i in (base.0)..self.stack.len() {
            if let Some(val) = self.get_stack(StackId::from(i)) {
                match *val {
                    Value::VarArgs => {
                        symbols.push(RegConst::VarArgs);
                        break;
                    }
                    Value::Return(call, true) => {
                        symbols.push(RegConst::VarCall(call));
                        break;
                    }
                    _ => symbols.push(RegConst::Stack(StackId::from(i))),
                }
            } else {
                break;
            }
        }
        symbols
    }

    // Set most recent value on a stack variable
    pub fn set_stack(&mut self, idx: StackId, val: Value) {
        if idx.0 >= self.stack.len() {
            self.stack.resize(idx.0 + 1, None);
        }
        self.stack[idx.0] = Some(val.clone());
        self.add_modified([idx]);
        self.operations.push(Operation::SetStack(idx, val));
    }

    // Get most recent value set on stack
    pub fn get_stack(&self, idx: StackId) -> Option<&Value> {
        self.stack.get(idx.0).unwrap_or(&None).as_ref()
    }

    pub fn table_mut(&mut self, id: TableId) -> Option<&mut Table> {
        self.tables.get_mut(id.0)
    }

    pub fn operation(&mut self, id: OperationId) -> Option<&Operation> {
        self.operations.get(id.0)
    }

    // Add symbol to ir history
    /*
    pub fn add_symbol(&mut self, symbol: SymbolRef) -> SymbolRef {
        self.symbols.push(symbol.clone());
        symbol
    }

    pub fn add(&mut self, dst: StackId, left: SymbolRef, right: SymbolRef) {
        let sum = self.add_symbol(Symbol::add(left, right));
        self.set_stack(dst, sum);
    }*/

    pub fn add(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        self.add_referenced(&[left, right]);
        self.set_stack(dst, Value::Add { left, right });
    }

    pub fn sub(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        self.add_referenced(&[left, right]);
        self.set_stack(dst, Value::Sub { left, right });
    }

    pub fn pow(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        self.add_referenced(&[left, right]);
        self.set_stack(dst, Value::Pow { left, right });
    }

    pub fn mul(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        self.add_referenced(&[left, right]);
        self.set_stack(dst, Value::Mul { left, right });
    }

    pub fn div(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        self.add_referenced(&[left, right]);
        self.set_stack(dst, Value::Div { left, right });
    }

    pub fn modulus(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        self.add_referenced(&[left, right]);
        self.set_stack(dst, Value::Mod { left, right });
    }

    pub fn number(&mut self, dst: StackId, n: f32) {
        self.set_stack(dst, Value::Number(n));
    }

    pub fn load_constant(&mut self, dst: StackId, constant: ConstantId) {
        self.set_stack(dst, Value::Symbol(RegConst::Constant(constant)));
    }

    // Makes a call. If param_count is None, the arguments are vararg
    // if return_count is None, the call is multiret
    pub fn call(
        &mut self,
        func: RegConst,
        param_base: StackId,
        param_count: Option<usize>,
        return_base: StackId,
        return_count: Option<usize>,
    ) -> OperationId {
        self.add_referenced(&[func]);
        let params = match param_count {
            None => {
                let mut p = Vec::new();
                let mut current = param_base.0;
                // Add values on stack until we find a vararg
                let mut found_va = false;
                for offset in param_base.0..self.stack.len() {
                    let val = self.get_stack(StackId::from(offset));
                    p.push(RegConst::Stack(StackId::from(offset)));
                    if matches!(val, Some(Value::VarArgs | Value::Return(_, true))) {
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
                    let s = RegConst::Stack(param_base + p);
                    s
                })
                .collect(),
        };
        self.add_referenced(&params);

        let op_id = OperationId(self.operations.len());

        let returns = match return_count {
            None => {
                self.set_stack(return_base, Value::Return(op_id, true));
                [RegConst::Stack(return_base)].to_vec()
            }
            Some(count) => (0..count)
                .map(|i| {
                    let val = Value::Return(op_id, false);
                    let return_pos = return_base + i;
                    self.set_stack(return_pos, val);
                    RegConst::Stack(return_pos)
                })
                .collect(),
        };

        let c = Operation::Call {
            func,
            params,
            returns,
            is_multiret: return_count.is_none(),
        };
        self.operations.push(c);
        op_id
    }

    pub fn closure(&mut self, dst: StackId, index: usize, upvalues: Vec<RegConst>) {
        self.add_referenced(&upvalues);
        let val = Value::Closure { index, upvalues };
        self.set_stack(dst, val);
    }

    pub fn concat(&mut self, dst: StackId, values: Vec<RegConst>) {
        self.add_referenced(&values);
        let val = Value::Concat(values);
        self.set_stack(dst, val);
    }

    pub fn get_table(&mut self, dst: StackId, table: RegConst, key: RegConst) {
        self.add_referenced(&[table, key]);
        let val = Value::GetTable { table, key };
        self.set_stack(dst, val);
    }

    pub fn set_table(&mut self, table: RegConst, key: RegConst, value: RegConst) {
        self.add_referenced(&[table, key, value]);
        self.operations
            .push(Operation::SetTable { table, key, value });
    }

    pub fn get_global(&mut self, dst: StackId, key: ConstantId) {
        self.set_stack(dst, Value::Global(key));
    }

    pub fn set_global(&mut self, key: ConstantId, val: RegConst) {
        self.add_referenced(&[val]);
        self.operations.push(Operation::SetGlobal(key, val));
    }

    pub fn set_cglobal(&mut self, key: ConstantId, val: RegConst) {
        self.add_referenced(&[val]);
        self.operations.push(Operation::SetCGlobal(key, val));
    }

    pub fn get_varargs(&mut self, dst: StackId, count: Option<usize>) {
        match count {
            None => {
                self.set_stack(dst, Value::VarArgs);
            }
            Some(count) => {
                let op_id = OperationId(self.operations.len());
                let args = (0..count)
                    .map(|i| {
                        let pos = dst + i;
                        self.set_stack(pos, Value::Arg(op_id));
                        RegConst::Stack(pos)
                    })
                    .collect();
                self.operations.push(Operation::GetVarArgs(args));
            }
        }
    }

    pub fn new_table(&mut self, dst: StackId) {
        let table_id = TableId(self.tables.len());
        self.tables.push(Table(Vec::new()));
        self.set_stack(dst, Value::Table(table_id));
    }

    pub fn set_list(&mut self, table: StackId, offset: usize, count: usize) {
        self.add_referenced(&[RegConst::Stack(table)]);

        let symbols: Vec<RegConst> = match count {
            0 => {
                // Search stack for vararg
                let mut symbols = self.base_to_vararg(table + 1);
                // Reverse for correct ordering
                symbols.reverse();
                symbols
            }
            _ => (0..count)
                .rev()
                .map(|i| RegConst::Stack(table + i))
                .collect(),
        };
        assert!(!symbols.is_empty());
        self.add_referenced(&symbols);

        let op = Operation::SetList(table, offset, symbols);
    }

    pub fn set_upvalue(&mut self, upvalue: UpvalueId, value: RegConst) {
        self.add_referenced(&[value]);
        self.operations.push(Operation::SetUpvalue(upvalue, value));
    }

    pub fn not(&mut self, dst: StackId, src: RegConst) {
        self.add_referenced(&[src]);
        self.set_stack(dst, Value::Not(src));
    }

    pub fn len(&mut self, dst: StackId, src: RegConst) {
        self.add_referenced(&[src]);
        self.set_stack(dst, Value::Len(src));
    }

    pub fn unm(&mut self, dst: StackId, src: RegConst) {
        self.add_referenced(&[src]);
        self.set_stack(dst, Value::Unm(src));
    }

    pub fn get_upvalue(&mut self, dst: StackId, id: UpvalueId) {
        self.set_stack(dst, Value::Upvalue(id));
    }

    pub fn load_boolean(&mut self, dst: StackId, value: bool) {
        self.set_stack(dst, Value::Boolean(value));
    }

    pub fn mov(&mut self, dst: StackId, src: StackId) {
        let src = RegConst::Stack(src);
        self.add_referenced(&[src]);
        self.set_stack(dst, Value::Symbol(src));
    }

    pub fn load_nil(&mut self, dst: StackId) {
        self.set_stack(dst, Value::Nil);
    }

    // Returns symbol of forloop index
    pub fn tail_forloop(
        &mut self,
        step: RegConst,
        limit: RegConst,
        init: RegConst,
        idx: StackId,
        inner: usize,
        end: usize,
    ) {
        self.add_referenced(&[step, limit, init]);
        self.set_stack(idx, Value::ForIndex);
        self.tail = Tail::ForLoop {
            init,
            limit,
            step,
            idx,
            inner,
            end,
        };
    }

    pub fn tail_eq(
        &mut self,
        left: RegConst,
        right: RegConst,
        direction: bool,
        target_1: usize,
        target_2: usize,
    ) {
        self.add_referenced(&[left, right]);

        self.tail = Tail::Eq(ConditionalB {
            left,
            right,
            direction,
            target_1,
            target_2,
        });
    }

    pub fn tail_lt(
        &mut self,
        left: RegConst,
        right: RegConst,
        direction: bool,
        target_1: usize,
        target_2: usize,
    ) {
        self.add_referenced(&[left, right]);

        self.tail = Tail::Lt(ConditionalB {
            left,
            right,
            direction,
            target_1,
            target_2,
        });
    }

    pub fn tail_le(
        &mut self,
        left: RegConst,
        right: RegConst,
        direction: bool,
        target_1: usize,
        target_2: usize,
    ) {
        self.add_referenced(&[left, right]);

        self.tail = Tail::Le(ConditionalB {
            left,
            right,
            direction,
            target_1,
            target_2,
        });
    }

    pub fn tail_testset(
        &mut self,
        test: RegConst,
        dst: StackId,
        direction: bool,
        target_1: usize,
        target_2: usize,
    ) {
        self.add_referenced(&[test, RegConst::Stack(dst)]);
        self.add_modified([dst]);

        self.tail = Tail::TestSet(
            ConditionalA {
                value: test,
                direction,
                target_1,
                target_2,
            },
            dst,
        );
    }

    pub fn tail_test(&mut self, test: RegConst, direction: bool, target_1: usize, target_2: usize) {
        self.add_referenced(&[test]);

        self.tail = Tail::Test(ConditionalA {
            value: test,
            direction,
            target_1,
            target_2,
        });
    }

    pub fn tail_tailcall(&mut self, func: StackId, nparams: Option<usize>) {
        let call = self.call(RegConst::Stack(func), func + 1, nparams, func, None);
        self.tail = Tail::TailCall(call);
    }

    pub fn tail_tforloop(
        &mut self,
        base: StackId,
        nresults: Option<usize>,
        inner: usize,
        end: usize,
    ) {
        let call = self.call(RegConst::Stack(base), base + 1, Some(2), base + 3, nresults);
        self.tail = Tail::TForLoop {
            call,
            index: RegConst::Stack(base + 1),
            state: RegConst::Stack(base + 2),
            inner,
            end,
        };
    }

    pub fn tail_return(&mut self, base: StackId, nresults: Option<usize>) {
        let results = match nresults {
            None => self.base_to_vararg(base),
            Some(count) => (0..count).map(|j| RegConst::Stack(base + j)).collect(),
        };
        self.add_referenced(&results);
        self.tail = Tail::Return(results);
    }

    pub fn var_label(&self, var: VariableId) -> String {
        format!("var{}", var.0)
    }

    pub fn call_src(&self, call: OperationId) -> String {
        let mut buff = Vec::new();
        if let Operation::Call {
            func,
            params,
            returns,
            is_multiret,
        } = &self.operations[call.0]
        {
            write!(&mut buff, "{}(", self.rc_label(*func)).unwrap();
            write!(
                &mut buff,
                "{})",
                params
                    .iter()
                    .map(|p| self.rc_label(*p))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
        }
        String::from_utf8(buff).unwrap()
    }

    pub fn rc_label(&self, rc: RegConst) -> String {
        match &rc {
            RegConst::Stack(id) => self.var_label(self.variables[id]),
            RegConst::Constant(cid) => format!("const{}", cid.0),
            RegConst::VarArgs => "...".to_string(),
            RegConst::VarCall(op) => self.call_src(*op),
        }
    }

    pub fn print_value(&self, value: &Value) {
        match value {
            Value::None => todo!(),
            Value::Symbol(rc) => print!("{}", self.rc_label(*rc)),
            Value::Number(n) => print!("{}", n),
            Value::Boolean(b) => print!("{}", b),
            Value::Param => todo!(),
            Value::Add { left, right } => {
                print!("{} + {}", self.rc_label(*left), self.rc_label(*right))
            }
            Value::Sub { left, right } => {
                print!("{} - {}", self.rc_label(*left), self.rc_label(*right))
            }
            Value::Div { left, right } => {
                print!("{} / {}", self.rc_label(*left), self.rc_label(*right))
            }
            Value::Mul { left, right } => {
                print!("{} * {}", self.rc_label(*left), self.rc_label(*right))
            }
            Value::Mod { left, right } => {
                print!("{} % {}", self.rc_label(*left), self.rc_label(*right))
            }
            Value::Pow { left, right } => {
                print!("{} ^ {}", self.rc_label(*left), self.rc_label(*right))
            }
            Value::Nil => print!("nil"),
            Value::Not(rc) => print!("not {}", self.rc_label(*rc)),
            Value::Unm(rc) => print!("-{}", self.rc_label(*rc)),
            Value::Len(rc) => print!("#{}", self.rc_label(*rc)),
            Value::Return(_, _) => {}
            Value::GetTable { table, key } => {
                print!("{}[{}]", self.rc_label(*table), self.rc_label(*key))
            }
            Value::Closure { index, upvalues } => todo!(),
            Value::Table(tid) => todo!(),
            Value::Upvalue(_) => todo!(),
            Value::ForIndex => todo!(),
            Value::Global(cid) => print!("const{}", cid.0),
            Value::VarArgs => print!("..."),
            Value::Arg(_) => todo!(),
            Value::Concat(rcs) => print!(
                "{}",
                rcs.iter()
                    .map(|rc| self.rc_label(*rc))
                    .collect::<Vec<String>>()
                    .join(" .. ")
            ),
            Value::Unknown(_) => todo!(),
            Value::ResolvedUnknown(_) => todo!(),
        };
    }

    pub fn print(&self) {
        for (op_id, op) in self.operations.iter().enumerate() {
            match op {
                Operation::SetStack(stack_id, val) => {
                    if val.is_var() || matches!(*val, Value::Return(_, _)) {
                        continue;
                    }
                    let var = self.variables[stack_id];
                    print!("{} = ", self.var_label(var));
                    self.print_value(val);
                    println!();
                }
                Operation::Call {
                    func,
                    params,
                    returns,
                    is_multiret,
                } => {
                    if *is_multiret {
                        continue;
                    }
                    if !returns.is_empty() {
                        print!(
                            "local {} = ",
                            returns
                                .iter()
                                .map(|p| self.rc_label(*p))
                                .collect::<Vec<String>>()
                                .join(", ")
                        );
                    }
                    println!("{}", self.call_src(OperationId(op_id)));
                }
                Operation::SetGlobal(_, _) => todo!(),
                Operation::SetCGlobal(_, _) => todo!(),
                Operation::SetUpvalue(_, _) => todo!(),
                Operation::SetTable { table, key, value } => todo!(),
                Operation::GetVarArgs(_) => todo!(),
                Operation::SetList(_, _, _) => todo!(),
            }
        }
    }
}



pub struct IrTree {
    nodes: HashMap<usize, IrNode>,
    closures: Vec<IrTree>,
}