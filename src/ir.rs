use std::{
    cell::RefCell,
    collections::{hash_map::Entry, HashMap, HashSet},
    io::Cursor,
    io::Write,
    ops::Add,
    rc::{Rc, Weak},
};

use crate::function::{Constant, Function, LvmInstruction};

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Symbol(VarConst),
    Number(f32),
    Boolean(bool),
    Param,
    Add {
        left: VarConst,
        right: VarConst,
    },
    Sub {
        left: VarConst,
        right: VarConst,
    },
    Div {
        left: VarConst,
        right: VarConst,
    },
    Mul {
        left: VarConst,
        right: VarConst,
    },
    Mod {
        left: VarConst,
        right: VarConst,
    },
    Pow {
        left: VarConst,
        right: VarConst,
    },
    Nil,
    Not(VarConst),
    Unm(VarConst),
    Len(VarConst),
    Return(OperationId, bool),
    GetTable {
        table: VarConst,
        key: VarConst,
    },
    Closure {
        index: usize,
        upvalues: Vec<VarConst>,
    },
    Table(TableId),
    //Table { items: Vec<Option<Symbol>> },
    Upvalue(UpvalueId),
    ForIndex,
    Global(ConstantId),
    VarArgs,
    Arg(OperationId),
    Concat(Vec<VarConst>),
}

impl Value {
    pub fn is_var(&self) -> bool {
        matches!(*self, Value::VarArgs | Value::Return(_, true))
    }
}

#[derive(Debug)]
pub enum Operation {
    SetStack(VariableRef, Value),
    Call {
        func: VarConst,
        params: Vec<VarConst>,
        returns: Vec<VarConst>,
        is_multiret: bool,
    },
    SetGlobal(ConstantId, VarConst),
    SetCGlobal(ConstantId, VarConst),
    SetUpvalue(UpvalueId, VarConst),
    SetTable {
        table: VarConst,
        key: VarConst,
        value: VarConst,
    },
    GetVarArgs(Vec<VarConst>),
    SetList(VariableRef, usize, Vec<VarConst>),
}

#[derive(Debug)]
pub struct Table(Vec<Option<VarConst>>);

// IR Symbol
#[derive(Debug, Clone, Copy)]
pub enum RegConst {
    Stack(StackId),
    UpValue(UpvalueId),
    Constant(ConstantId),
}

// IR Symbol
#[derive(Debug, Clone)]
pub enum VarConst {
    Var(VariableRef),
    UpValue(UpvalueId),
    Constant(ConstantId),
    VarArgs,
    VarCall(OperationId),
}

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
// Id of an upvalue
pub struct UpvalueId(pub usize);

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
    pub value: VarConst,
    pub direction: bool,
    pub target_1: usize,
    pub target_2: usize,
}

#[derive(Debug)]
pub struct ConditionalB {
    pub left: VarConst,
    pub right: VarConst,
    pub direction: bool,
    pub target_1: usize,
    pub target_2: usize,
}

#[derive(Debug)]
pub enum Tail {
    None,
    Return(Vec<VarConst>),
    TailCall(OperationId),
    Eq(ConditionalB),
    Le(ConditionalB),
    Lt(ConditionalB),
    TestSet(ConditionalA, VariableRef),
    Test(ConditionalA),
    TForLoop {
        call: OperationId,
        index: VariableRef,
        state: VariableRef,
        inner: usize,
        end: usize,
    },
    ForLoop {
        init: VarConst,
        limit: VarConst,
        step: VarConst,
        idx: VariableRef,
        inner: usize,
        end: usize,
    },
}

#[derive(Debug)]
pub struct Variable {
    pub references: Vec<VariableRef>,
    pub last_value: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a variable
pub struct VariableId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// Id of a variable reference
pub struct VariableRef(pub usize);

#[derive(Debug)]
pub struct VariableSolver {
    pub variables: Vec<Variable>,
    pub references: Vec<VariableId>,
}

impl VariableSolver {
    pub fn new() -> VariableSolver {
        VariableSolver {
            variables: Vec::new(),
            references: Vec::new(),
        }
    }

    fn minimize_impl(
        &mut self,
        tree: &IrTree,
        reg: StackId,
        node: usize,
        to: &VariableRef,
        checked: &mut HashSet<usize>,
    ) {
        if !checked.insert(node) {
            return;
        }
        if let Some(v) = tree.nodes[&node].variables.get(&reg) {
            self.combine(to, v);
            return;
        }

        for n in &tree.prev[&node] {
            /*if *n == usize::MAX {
                continue;
            }*/
            self.minimize_impl(tree, reg, *n, to, checked);
        }
    }

    pub fn minimize(&mut self, tree: &IrTree) {
        for (nid, n) in &tree.nodes {
            for (sid, vref) in &n.references {
                self.minimize_impl(tree, *sid, *nid, vref, &mut HashSet::new());
            }
        }
    }

    pub fn combine(&mut self, to: &VariableRef, from: &VariableRef) {
        let to = self.references[to.0];
        let from = self.references[from.0];
        if let Some(lv) = &self.variables[from.0].last_value {
            self.variables[to.0].last_value = Some(lv.clone());
        }
        let to_combine = std::mem::replace(&mut self.variables[from.0].references, Vec::new());
        self.variables[to.0]
            .references
            .extend(to_combine.iter().map(|x| x.clone()));
        for reference in to_combine {
            self.references[reference.0] = to;
        }
    }

    pub fn reference(&mut self, variable: &VariableRef) -> VariableRef {
        let variable = self.references[variable.0];
        let var_ref = VariableRef(self.references.len());
        self.variables[variable.0].references.push(var_ref.clone());
        self.references.push(variable);
        var_ref
    }

    pub fn new_variable(&mut self) -> VariableRef {
        let var_id = VariableId(self.variables.len());
        let ref_id = VariableRef(self.references.len());
        self.variables.push(Variable {
            references: Vec::from_iter([ref_id.clone()]),
            last_value: None,
        });
        self.references.push(var_id);
        ref_id
    }

    pub fn set_last_value(&mut self, var: &VariableRef, value: Value) {
        self.variables[self.references[var.0].0].last_value = Some(value);
    }

    pub fn last_value(&self, var: &VariableRef) -> Option<&Value> {
        self.variables[self.references[var.0].0].last_value.as_ref()
    }

    pub fn get_variable(&self, var: &VariableRef) -> &Variable {
        &self.variables[self.references[var.0].0]
    }
}

// Context of IR instructions
#[derive(Debug)]
pub struct IrNode {
    // Array of all symbols generated in this context
    pub operations: Vec<Operation>,
    pub tables: Vec<Table>,
    // Map of register to variable
    pub variables: HashMap<StackId, VariableRef>,
    pub references: HashMap<StackId, VariableRef>,
    pub tail: Tail,
}

#[derive(Debug)]
pub struct IrNodeBuilder<'a> {
    pub solver: &'a mut VariableSolver,
    // The most recent operations that operated on the stack
    pub stack: Vec<Option<Value>>,
    // Array of all symbols generated in this context
    pub operations: Vec<Operation>,
    pub tables: Vec<Table>,
    // Map of register to variable
    pub variables: HashMap<StackId, VariableRef>,
    pub references: HashMap<StackId, VariableRef>,
    pub tail: Tail,
}

impl IrNodeBuilder<'_> {
    pub fn new(solver: &mut VariableSolver) -> IrNodeBuilder {
        IrNodeBuilder {
            solver,
            stack: Vec::new(),
            operations: Vec::new(),
            tables: Vec::new(),
            variables: HashMap::new(),
            references: HashMap::new(),
            tail: Tail::None,
        }
    }

    pub fn build(self) -> IrNode {
        IrNode {
            operations: self.operations,
            tables: self.tables,
            variables: self.variables,
            references: self.references,
            tail: self.tail,
        }
    }

    pub fn reference_stack(&mut self, id: StackId) -> VariableRef {
        if let Some(var) = self.variables.get(&id) {
            self.solver.reference(var)
        } else {
            match self.references.entry(id) {
                Entry::Occupied(occ) => self.solver.reference(occ.get()),
                Entry::Vacant(vac) => vac.insert(self.solver.new_variable()).clone(),
            }
        }
    }

    pub fn reference_regconst(&mut self, rc: RegConst) -> VarConst {
        match rc {
            RegConst::Stack(id) => VarConst::Var(self.reference_stack(id)),
            RegConst::UpValue(id) => VarConst::UpValue(id),
            RegConst::Constant(id) => VarConst::Constant(id),
        }
    }

    pub fn modify_stack(&mut self, id: StackId) -> VariableRef {
        let var = self.solver.new_variable();
        self.variables.insert(id, var.clone());
        var
    }

    // Get last stack symbols from base to the first vararg
    fn base_to_vararg(&mut self, base: StackId) -> Vec<VarConst> {
        let mut symbols = Vec::new();
        for i in (base.0)..self.stack.len() {
            if let Some(val) = self.get_stack(StackId::from(i)) {
                match *val {
                    Value::VarArgs => {
                        symbols.push(VarConst::VarArgs);
                        break;
                    }
                    Value::Return(call, true) => {
                        symbols.push(VarConst::VarCall(call));
                        break;
                    }
                    _ => symbols.push(VarConst::Var(self.reference_stack(StackId::from(i)))),
                }
            } else {
                break;
            }
        }
        symbols
    }

    // Set most recent value on a stack variable
    pub fn set_stack(&mut self, idx: StackId, val: Value, add_op: bool) -> VariableRef {
        if idx.0 >= self.stack.len() {
            self.stack.resize(idx.0 + 1, None);
        }
        self.stack[idx.0] = Some(val.clone());
        let vref = self.modify_stack(idx);
        self.solver.set_last_value(&vref, val.clone());
        if add_op {
            let op = Operation::SetStack(vref.clone(), val);
            self.operations.push(op);
        }
        vref
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
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);
        self.set_stack(dst, Value::Add { left, right }, true);
    }

    pub fn sub(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);
        self.set_stack(dst, Value::Sub { left, right }, true);
    }

    pub fn pow(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);
        self.set_stack(dst, Value::Pow { left, right }, true);
    }

    pub fn mul(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);
        self.set_stack(dst, Value::Mul { left, right }, true);
    }

    pub fn div(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);
        self.set_stack(dst, Value::Div { left, right }, true);
    }

    pub fn modulus(&mut self, dst: StackId, left: RegConst, right: RegConst) {
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);
        self.set_stack(dst, Value::Mod { left, right }, true);
    }

    pub fn number(&mut self, dst: StackId, n: f32) {
        self.set_stack(dst, Value::Number(n), true);
    }

    pub fn load_constant(&mut self, dst: StackId, constant: ConstantId) {
        self.set_stack(dst, Value::Symbol(VarConst::Constant(constant)), true);
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
        let func = self.reference_regconst(func);
        let params = match param_count {
            None => {
                let mut p = Vec::new();
                let mut current = param_base.0;
                // Add values on stack until we find a vararg
                let mut found_va = false;
                for offset in param_base.0..self.stack.len() {
                    p.push(VarConst::Var(self.reference_stack(StackId::from(offset))));
                    let val = self.get_stack(StackId::from(offset));
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
                    let s = VarConst::Var(self.reference_stack(param_base + p));
                    s
                })
                .collect(),
        };

        let op_id = OperationId(self.operations.len());

        let returns = match return_count {
            None => {
                let vref = self.set_stack(return_base, Value::Return(op_id, true), false);
                [VarConst::Var(vref)].to_vec()
            }
            Some(count) => (0..count)
                .map(|i| {
                    let val = Value::Return(op_id, false);
                    let return_pos = return_base + i;
                    let vref = self.set_stack(return_pos, val, false);
                    VarConst::Var(vref)
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
        let upvalues = upvalues
            .into_iter()
            .map(|v| self.reference_regconst(v))
            .collect();
        let val = Value::Closure { index, upvalues };
        self.set_stack(dst, val, true);
    }

    pub fn concat(&mut self, dst: StackId, values: Vec<RegConst>) {
        let values = values
            .into_iter()
            .map(|v| self.reference_regconst(v))
            .collect();
        let val = Value::Concat(values);
        self.set_stack(dst, val, true);
    }

    pub fn get_table(&mut self, dst: StackId, table: RegConst, key: RegConst) {
        let table = self.reference_regconst(table);
        let key = self.reference_regconst(key);
        let val = Value::GetTable { table, key };
        self.set_stack(dst, val, true);
    }

    pub fn set_table(&mut self, table: RegConst, key: RegConst, value: RegConst) {
        let table = self.reference_regconst(table);
        let key = self.reference_regconst(key);
        let value = self.reference_regconst(value);
        self.operations
            .push(Operation::SetTable { table, key, value });
    }

    pub fn get_global(&mut self, dst: StackId, key: ConstantId) {
        self.set_stack(dst, Value::Global(key), true);
    }

    pub fn set_global(&mut self, key: ConstantId, val: RegConst) {
        let val = self.reference_regconst(val);
        self.operations.push(Operation::SetGlobal(key, val));
    }

    pub fn set_cglobal(&mut self, key: ConstantId, val: RegConst) {
        let val = self.reference_regconst(val);
        self.operations.push(Operation::SetCGlobal(key, val));
    }

    pub fn get_varargs(&mut self, dst: StackId, count: Option<usize>) {
        match count {
            None => {
                self.set_stack(dst, Value::VarArgs, true);
            }
            Some(count) => {
                let op_id = OperationId(self.operations.len());
                let args = (0..count)
                    .map(|i| {
                        let pos = dst + i;
                        self.set_stack(pos, Value::Arg(op_id), false);
                        VarConst::Var(self.modify_stack(pos))
                    })
                    .collect();
                self.operations.push(Operation::GetVarArgs(args));
            }
        }
    }

    pub fn new_table(&mut self, dst: StackId) {
        let table_id = TableId(self.tables.len());
        self.tables.push(Table(Vec::new()));
        self.set_stack(dst, Value::Table(table_id), true);
    }

    pub fn set_list(&mut self, table: StackId, offset: usize, count: usize) {
        let table_ref = self.reference_stack(table);

        let symbols: Vec<VarConst> = match count {
            0 => {
                // Search stack for vararg
                let mut symbols = self.base_to_vararg(table + 1_usize);
                // Reverse for correct ordering
                symbols.reverse();
                symbols
            }
            _ => (0..count)
                .rev()
                .map(|i| VarConst::Var(self.reference_stack(table + i)))
                .collect(),
        };
        assert!(!symbols.is_empty());

        let op = Operation::SetList(table_ref, offset, symbols);
    }

    pub fn set_upvalue(&mut self, upvalue: UpvalueId, value: RegConst) {
        let value = self.reference_regconst(value);
        self.operations.push(Operation::SetUpvalue(upvalue, value));
    }

    pub fn not(&mut self, dst: StackId, src: RegConst) {
        let src = self.reference_regconst(src);
        self.set_stack(dst, Value::Not(src), true);
    }

    pub fn len(&mut self, dst: StackId, src: RegConst) {
        let src = self.reference_regconst(src);
        self.set_stack(dst, Value::Len(src), true);
    }

    pub fn unm(&mut self, dst: StackId, src: RegConst) {
        let src = self.reference_regconst(src);
        self.set_stack(dst, Value::Unm(src), true);
    }

    pub fn get_upvalue(&mut self, dst: StackId, id: UpvalueId) {
        self.set_stack(dst, Value::Upvalue(id), true);
    }

    pub fn load_boolean(&mut self, dst: StackId, value: bool) {
        self.set_stack(dst, Value::Boolean(value), true);
    }

    pub fn mov(&mut self, dst: StackId, src: StackId) {
        let src = RegConst::Stack(src);
        let src = self.reference_regconst(src);
        self.set_stack(dst, Value::Symbol(src), true);
    }

    pub fn load_nil(&mut self, dst: StackId) {
        self.set_stack(dst, Value::Nil, true);
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
        let step = self.reference_regconst(step);
        let limit = self.reference_regconst(limit);
        let init = self.reference_regconst(init);
        self.set_stack(idx, Value::ForIndex, false);
        self.tail = Tail::ForLoop {
            init,
            limit,
            step,
            idx: self.modify_stack(idx),
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
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);

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
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);

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
        let left = self.reference_regconst(left);
        let right = self.reference_regconst(right);

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
        self.reference_stack(dst);
        let test = self.reference_regconst(test);
        let dst = self.modify_stack(dst);

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
        let test = self.reference_regconst(test);

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
            index: self.reference_stack(base + 1),
            state: self.reference_stack(base + 2),
            inner,
            end,
        };
    }

    pub fn tail_return(&mut self, base: StackId, nresults: Option<usize>) {
        let results = match nresults {
            None => self.base_to_vararg(base),
            Some(count) => (0..count)
                .map(|j| VarConst::Var(self.reference_stack(base + j)))
                .collect(),
        };
        self.tail = Tail::Return(results);
    }

    /*
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
            RegConst::UpValue(_) => todo!(),
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
    }*/
}

#[derive(Debug)]
pub struct IrTree<'a> {
    pub nodes: HashMap<usize, IrNode>,
    pub next: HashMap<usize, Vec<usize>>,
    pub prev: HashMap<usize, Vec<usize>>,
    pub func: &'a Function,
    pub closures: Vec<IrTree<'a>>,
    pub statics: HashSet<StackId>,
}

impl IrTree<'_> {
    pub fn new(func: &Function) -> IrTree {
        IrTree {
            nodes: HashMap::new(),
            next: HashMap::new(),
            prev: HashMap::new(),
            func,
            closures: Vec::new(),
            statics: HashSet::new(),
        }
    }

    pub fn connect_node(&mut self, prev: usize, next: usize) {
        self.next.entry(prev).or_default().push(next);
        self.prev.entry(next).or_default().push(prev);
    }

    pub fn add_node(&mut self, id: usize, node: IrNode) {
        self.nodes.insert(id, node);
        // Ensure next/prev entries exist
        self.next.entry(id).or_default();
        self.prev.entry(id).or_default();
    }

    pub fn add_static(&mut self, id: StackId) {
        self.statics.insert(id);
    }
}
