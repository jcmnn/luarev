use std::{
    borrow::BorrowMut,
    cell::{Cell, Ref, RefCell},
    collections::{HashMap, HashSet},
    ops::Add,
    rc::{Rc, Weak},
};

use int_enum::IntEnumError;
use thiserror::Error;

use crate::{
    function::{Constant, Function, LvmInstruction, Name, OpCode},
    ir::{
        ConstantId, IrContext, OperationId, RegConst, StackId, Tail, UpvalueId, Value, VariableId,
    },
};

#[derive(Debug, Error)]
pub enum DecompileError {
    #[error("invalid opcode {0}")]
    OpCodeError(#[from] IntEnumError<OpCode>),
    #[error("unexpected end of instructions")]
    UnexpectedEnd,
    #[error("invalid upvalue")]
    InvalidUpvalue,
    #[error("expected JMP instruction after branch")]
    ExpectedJmp,
    #[error("expected NewTable in SETLIST instruction")]
    ExpectedTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Id of a node
pub struct NodeId(usize);

// Graph node
#[derive(Debug)]
pub struct Node {
    id: NodeId,
    offset: usize,
    last_offset: usize,
    ir: IrContext,
    next: Vec<NodeId>,
    prev: Vec<NodeId>,
}

#[derive(Debug)]
pub struct Variable {
    id: VariableId,
    register: StackId,
    references: Vec<NodeId>,
    modifies: Vec<NodeId>,
}

// Returns true if encoded value contains a constant index
const fn isk(x: u32) -> bool {
    (x & (1 << 8)) != 0
}

// Returns constant index from encoded value
const fn indexk(x: u32) -> usize {
    (x & !(1 << 8)) as usize
}

impl Node {
    fn new(id: NodeId, offset: usize) -> Node {
        Node {
            id,
            offset,
            last_offset: offset,
            ir: IrContext::new(),
            next: Vec::new(),
            prev: Vec::new(),
        }
    }

    fn ends_at(&self, end: &Rc<Node>) -> bool {
        false
    }

    fn add_next(&mut self, next: NodeId) {
        self.next.push(next);
    }

    fn add_prev(&mut self, prev: NodeId) {
        self.prev.push(prev);
    }

    // Returns value on stack or constant from encoded value
    fn stack_or_const(&mut self, r: u32, ctx: &FunctionContext) -> RegConst {
        if isk(r) {
            return RegConst::Constant(ConstantId(indexk(r))); //(ctx.func.constants[indexk(r)].clone());
        }

        RegConst::Stack(StackId::from(r))
    }

    // Adds instructions to graph node
    fn add_code(
        &mut self,
        ctx: &mut FunctionContext,
        mut offset: usize,
    ) -> Result<(), DecompileError> {
        loop {
            let i = ctx.func.code[offset];

            // Decode instruction
            match i.opcode()? {
                OpCode::Closure => {
                    let bx = i.argbx() as usize;
                    if bx >= ctx.func.closures.len() {
                        println!("Invalid closure index");
                        break;
                    }
                    let closure = &ctx.func.closures[bx];
                    let upvalues = (0..closure.nups)
                        .map(|idx| {
                            let upi = ctx
                                .func
                                .code
                                .get(offset + 1 + idx as usize)
                                .ok_or(DecompileError::UnexpectedEnd)?;
                            match upi.opcode()? {
                                OpCode::GetUpval => Ok(ctx.upvalues[upi.argb() as usize].clone()),
                                OpCode::Move => {
                                    let reg = StackId::from(upi.argb());
                                    ctx.static_registers.insert(reg);
                                    Ok(RegConst::Stack(reg))
                                }
                                _ => Err(DecompileError::InvalidUpvalue),
                            }
                        })
                        .collect::<Result<Vec<RegConst>, DecompileError>>()?;

                    // Replace upvalues in closure
                    ctx.closures[bx].upvalues = upvalues; //.extend_from_slice(&upvalues);

                    self.ir.closure(
                        StackId::from(i.arga()),
                        bx,
                        ctx.closures[bx].upvalues.clone(),
                    );
                }
                OpCode::Div => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.div(StackId::from(i.arga()), left, right);
                }
                OpCode::GetNum => {
                    self.ir.number(StackId::from(i.arga()), i.number());
                }
                OpCode::Concat => {
                    let b = i.argb();
                    let c = i.argc();

                    let params: Vec<RegConst> =
                        (b..=c).map(|x| RegConst::Stack(StackId::from(x))).collect();

                    self.ir.concat(StackId::from(i.arga()), params);
                }
                OpCode::GetTable => {
                    let table = RegConst::Stack(StackId::from(i.argb()));
                    let key = self.stack_or_const(i.argc(), ctx);
                    self.ir.get_table(StackId::from(i.arga()), table, key);
                }
                OpCode::SetList => {
                    let mut n = i.argb() as usize;
                    let ra = i.arga() as usize;
                    let c = match i.argc() {
                        0 => {
                            let nexti = ctx
                                .func
                                .code
                                .get(offset + 1)
                                .ok_or(DecompileError::UnexpectedEnd)?;
                            nexti.raw() as usize
                        }
                        c => c as usize,
                    };
                    assert!(c != 0);

                    let table = StackId::from(ra);
                    self.ir.set_list(table, (c - 1) * 50, n);

                    /*
                    if n == 0 {
                        // Search stack for vararg
                        for j in (ra + 1)..self.ir.stack.len() {
                            let val = self.ir.get_stack(StackId::from(j));
                            if val.borrow().is_var() {
                                n = j - ra;
                                break;
                            }
                        }
                        assert!(n != 0);
                    }

                    let mut last = (((c - 1) * 50) + n) as usize;

                    // Resize table if needed
                    if last > items.len() {
                        items.resize(last, None);
                    }

                    for idx in (1..=n).rev() {
                        let val = self.ir.get_stack(StackId::from(ra + idx as usize));
                        RefCell::borrow_mut(&val).add_reference(&table);
                        items[last - 1] = Some(val);
                        last -= 1;
                    }*/
                }
                OpCode::LoadK => {
                    self.ir.load_constant(
                        StackId::from(i.arga()),
                        ConstantId(i.argbx() as usize),
                        // ctx.func.constants[i.argbx() as usize].clone(),
                    );
                }
                OpCode::SetGlobal => {
                    let cval = ConstantId(i.argbx() as usize); // ctx.func.constants[i.argbx() as usize].clone();
                    self.ir
                        .set_global(cval, RegConst::Stack(StackId::from(i.arga())));
                }
                OpCode::SetCGlobal => {
                    let key = ConstantId(i.argbx() as usize); // ctx.func.constants[i.argbx() as usize].clone();
                    self.ir
                        .set_cglobal(key, RegConst::Stack(StackId::from(i.arga())));
                }
                OpCode::Jmp => {}
                OpCode::TForLoop => {
                    let ra = StackId::from(i.arga());
                    let nresults = i.argc() as i32;

                    let nexti = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    let target = ((offset as i32) + 2 + nexti.argsbx()) as usize;

                    self.ir.tail_tforloop(
                        ra,
                        if nresults == -1 {
                            None
                        } else {
                            Some(nresults as usize)
                        },
                        target,
                        offset + 2,
                    );
                }
                OpCode::SetUpval => {
                    let src = RegConst::Stack(StackId::from(i.arga()));
                    self.ir.set_upvalue(UpvalueId::from(i.argb() as usize), src);
                }
                OpCode::Not => {
                    self.ir.not(
                        StackId::from(i.arga()),
                        RegConst::Stack(StackId::from(i.argb())),
                    );
                }
                OpCode::Vararg => {
                    let ra = i.arga();
                    let b = i.argb() as i32 - 1;
                    self.ir.get_varargs(
                        StackId::from(ra),
                        if b == -1 { None } else { Some(b as usize) },
                    )
                }
                OpCode::GetUpval => {
                    self.ir
                        .get_upvalue(StackId::from(i.arga()), UpvalueId::from(i.argb() as usize));
                }
                OpCode::Add => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.add(StackId::from(i.arga()), left, right);
                }
                OpCode::Return => {
                    let ra = StackId::from(i.arga());
                    let nresults = i.argb() as i32 - 1;
                    self.ir.tail_return(
                        ra,
                        if nresults == -1 {
                            None
                        } else {
                            Some(nresults as usize)
                        },
                    );
                }
                OpCode::GetGlobal => {
                    let key = ConstantId(i.argbx() as usize); // ctx.func.constants[i.argbx() as usize].clone();
                    self.ir.get_global(StackId::from(i.arga()), key);
                }
                OpCode::Len => {
                    self.ir.len(
                        StackId::from(i.arga()),
                        RegConst::Stack(StackId::from(i.argb())),
                    );
                }
                OpCode::Mul => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.mul(StackId::from(i.arga()), left, right);
                }
                OpCode::NewTable => {
                    self.ir.new_table(StackId::from(i.arga()));
                }
                OpCode::SetTable => {
                    let table = RegConst::Stack(StackId::from(i.arga()));
                    let key = self.stack_or_const(i.argb(), ctx);
                    let value = self.stack_or_const(i.argc(), ctx);
                    self.ir.set_table(table, key, value);
                }
                OpCode::Unm => {
                    self.ir.unm(
                        StackId::from(i.arga()),
                        RegConst::Stack(StackId::from(i.argb())),
                    );
                }
                OpCode::Mod => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.modulus(StackId::from(i.arga()), left, right);
                }
                OpCode::ForLoop => {
                    let ra = StackId::from(i.arga());
                    let step = RegConst::Stack(ra + 2_usize);
                    let limit = RegConst::Stack(ra + 1_usize);
                    let init = RegConst::Stack(ra);

                    let target = (offset as i32 + 1 + i.argsbx()) as usize;
                    self.ir
                        .tail_forloop(step, limit, init, ra + 3_usize, target, offset + 1)
                }
                OpCode::Call => {
                    let ra = i.arga() as usize;
                    let func = RegConst::Stack(StackId::from(ra));
                    let nparams = i.argb() as i32 - 1;
                    let nresults = i.argc() as i32 - 1;
                    self.ir.call(
                        func,
                        StackId::from(ra + 1),
                        if nparams == -1 {
                            None
                        } else {
                            Some(nparams as usize)
                        },
                        StackId::from(ra),
                        if nresults == -1 {
                            None
                        } else {
                            Some(nresults as usize)
                        },
                    );
                    /*
                    let results = match nresults {
                        -1 => {
                            let res = self.make_value(Value::VarArg);
                            self.stack.set(ra, res.clone());
                            [res].to_vec()
                        }
                        _ => (0..nresults as usize)
                            .map(|ri| {
                                let val = self.make_value(Value::ReturnValue(func.clone()));
                                self.stack.set(ra + 3 + ri, val.clone());
                                val
                            })
                            .collect(),
                    };

                    let params = (0..nparams)
                        .map(|pi| self.stack.get(ra + 1 + pi as usize))
                        .collect();
                    let _call = self.make_value(Value::Call(func, params, results));
                    */
                }
                OpCode::LoadBool => {
                    self.ir.load_boolean(StackId::from(i.arga()), i.argb() != 0);
                }
                OpCode::ForPrep => {
                    // We don't need to do anything here
                }
                OpCode::Pow => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.pow(StackId::from(i.arga()), left, right);
                }
                OpCode::OpSelf => {
                    // TODO: Dedicated self operation
                    let table = RegConst::Stack(StackId::from(i.argb()));
                    let key = self.stack_or_const(i.argc(), ctx);
                    self.ir
                        .set_stack(StackId::from(i.arga() + 1), Value::Symbol(table));
                    self.ir.get_table(StackId::from(i.arga()), table, key);
                }
                OpCode::Sub => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.sub(StackId::from(i.arga()), left, right);
                }
                OpCode::Move => {
                    self.ir
                        .mov(StackId::from(i.arga()), StackId::from(i.argb()));
                }
                OpCode::Close => {}
                OpCode::LoadNil => {
                    let ra = i.arga();
                    let rb = i.argb();
                    for ri in ra..=rb {
                        self.ir.load_nil(StackId::from(ri));
                    }
                }
                // Tails
                OpCode::Eq => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.add_referenced(&[left, right]);

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = offset as i32 + ijmp.argsbx() + 2;
                    self.ir
                        .tail_eq(left, right, i.arga() == 0, offset + 2, target as usize);
                }
                OpCode::Lt => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.add_referenced(&[left, right]);

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = offset as i32 + ijmp.argsbx() + 2;
                    self.ir
                        .tail_lt(left, right, i.arga() == 0, offset + 2, target as usize);
                }
                OpCode::Le => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.add_referenced(&[left, right]);

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = offset as i32 + ijmp.argsbx() + 2;
                    self.ir
                        .tail_le(left, right, i.arga() == 0, offset + 2, target as usize);
                }
                OpCode::TestSet => {
                    let test = RegConst::Stack(StackId::from(i.argb()));
                    let dst = StackId::from(i.arga());

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    self.ir.tail_testset(
                        test,
                        dst,
                        i.arga() == 0,
                        offset + 2,
                        (offset as i32 + ijmp.argsbx() + 2) as usize,
                    );
                }
                OpCode::Test => {
                    let test = RegConst::Stack(StackId::from(i.arga()));
                    self.ir.add_referenced(&[test]);

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    self.ir.tail_test(
                        test,
                        i.arga() == 0,
                        offset + 2,
                        (offset as i32 + ijmp.argsbx() + 2) as usize,
                    );
                }
                OpCode::TailCall => {
                    let ra = StackId::from(i.arga());
                    let nparams = i.argb() as i32 - 1;
                    let nresults = i.argc() as i32 - 1;
                    if nresults != -1 {
                        println!("Tail call is not multiret. LVM will throw");
                        // not multiret; LVM throws here
                    }
                    self.ir.tail_tailcall(
                        ra,
                        if nparams == -1 {
                            // Vararg
                            None
                        } else {
                            Some(nparams as usize)
                        },
                    );
                }
            }

            if ctx.branches[offset].len() == 1 {
                let next_offset = ctx.branches[offset][0];
                if ctx.references[next_offset].len() > 1 {
                    break;
                }
                offset = next_offset;
                assert!(matches!(self.ir.tail, Tail::None));
            } else {
                break;
            }
        }
        self.last_offset = offset;
        Ok(())
    }
}

#[derive(Debug)]
pub struct RootContext {
    local_idx: Cell<usize>,
}

impl RootContext {
    pub fn new() -> RootContext {
        RootContext {
            local_idx: Cell::new(0),
        }
    }

    pub fn make_local(&self) -> String {
        let idx = self.local_idx.get();
        self.local_idx.set(idx + 1);
        format!("local_{}", idx)
    }
}

struct NodeReferences {
    uses_from: HashMap<StackId, Vec<NodeId>>,
    used_by: HashMap<StackId, Vec<NodeId>>,
}

#[derive(Debug)]
pub struct FunctionContext {
    func: Rc<Function>,
    nodes: Vec<RefCell<Node>>,
    branches: Vec<Vec<usize>>,
    params: Vec<StackId>,
    references: Vec<Vec<usize>>,
    // List of registers that must have a static variable throughout the program
    static_registers: HashSet<StackId>,
    root: Rc<RootContext>,
    upvalues: Vec<RegConst>,
    closures: Vec<FunctionContext>,
}

#[derive(Debug)]
pub struct VariableContext {
    variables: Vec<Option<Variable>>,
}

impl VariableContext {
    fn new() -> VariableContext {
        VariableContext {
            variables: Vec::new(),
        }
    }
    fn make_variable(&mut self, id: NodeId, register: StackId) -> VariableId {
        let var_id = VariableId(self.variables.len());
        self.variables.push(Some(Variable {
            id: var_id,
            references: Vec::new(),
            modifies: [id].to_vec(),
            register,
        }));
        var_id
    }

    fn make_reference(&mut self, id: NodeId, register: StackId) -> VariableId {
        let var_id = VariableId(self.variables.len());
        self.variables.push(Some(Variable {
            id: var_id,
            references: [id].to_vec(),
            modifies: Vec::new(),
            register,
        }));
        var_id
    }

    fn combine(&mut self, dst: VariableId, src: VariableId) {
        let Variable {
            id: _,
            modifies,
            references,
            register: _,
        } = self.variables[src.0].take().unwrap();

        let dst = self.variables[dst.0].as_mut().unwrap();
        dst.modifies.extend(modifies);
        dst.references.extend(references);
    }
}

impl FunctionContext {
    fn new(root: Rc<RootContext>, func: Rc<Function>) -> FunctionContext {
        let branches = vec![Vec::new(); func.code.len()];
        let references = vec![Vec::new(); func.code.len()];
        FunctionContext {
            upvalues: Vec::new(), //(0..func.nups).map(|_| Symbol::upvalue()).collect(),
            closures: func
                .closures
                .iter()
                .map(|f| FunctionContext::new(root.clone(), f.clone()))
                .collect(),
            params: (0..func.num_params).map(|i| StackId::from(i)).collect(),
            func,
            nodes: Vec::new(),
            branches,
            references,
            static_registers: HashSet::new(),
            root,
        }
    }

    fn add_branch(&mut self, src: usize, dst: usize) -> Result<(), DecompileError> {
        if src >= self.branches.len() || dst >= self.references.len() {
            return Err(DecompileError::UnexpectedEnd);
        }

        self.branches[src].push(dst);
        self.references[dst].push(src);

        Ok(())
    }

    fn node_at(&self, offset: usize) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|&x| x.borrow().offset == offset)
            .map(|x| x.borrow().id)
    }

    fn propogate_variables(
        &self,
        node_id: NodeId,
        vc: &mut VariableContext,
        made: &mut HashSet<NodeId>,
    ) {
        if made.contains(&node_id) {
            return;
        }
        made.insert(node_id);

        let mut to_combine = Vec::new();
        {
            // Propgate all previous nodes first
            let node = &self.nodes[node_id.0];

            // Make local variables
            {
                let mut n = node.borrow_mut();
                let vars: HashMap<StackId, VariableId> = HashMap::from_iter(
                    n.ir.stack_modified
                        .iter()
                        .map(|m| (*m, vc.make_variable(node_id, *m))),
                );
                n.ir.variables = vars;

                let refs: HashMap<StackId, VariableId> = HashMap::from_iter(
                    n.ir.stack_references
                        .iter()
                        .map(|m| (*m, vc.make_reference(node_id, *m))),
                );
                n.ir.references = refs;
            }

            let mut refs: HashMap<StackId, VariableId> = node.borrow_mut().ir.references.clone();
            for i in &node.borrow().next {
                self.propogate_variables(*i, vc, made);
                // Add next node's references to this node
                if node_id != *i {
                    for (stack_id, var) in &self.nodes[i.0].borrow().ir.references {
                        if node.borrow().ir.stack_modified.contains(stack_id) {
                            to_combine.push((node.borrow().ir.variables[stack_id], *var, *stack_id));
                        } else if let Some(old) = refs.insert(*stack_id, *var) {
                            if old != *var {
                                to_combine.push((old, *var, *stack_id));
                            }
                        }
                    }
                }
            }

            let mut node = node.borrow_mut();
            node.ir.references = refs;
        }

        for (dst, src, stack_id) in to_combine.iter().rev() {
            // Combine references
            for id in &vc.variables[src.0].as_mut().unwrap().references {
                self.nodes[id.0].borrow_mut().ir.references.insert(*stack_id,*dst);
            }
            // Combine modified
            for id in &vc.variables[src.0].as_mut().unwrap().modifies {
                self.nodes[id.0].borrow_mut().ir.variables.insert(*stack_id,*dst);
            }
            vc.combine(*dst, *src);
        }
    }

    fn make_variables(&mut self, vc: &mut VariableContext) -> Result<(), DecompileError> {
        // Add static variables to root node
        {
            /*
            let mut root_node = self.nodes[0].borrow_mut();
            // Add function parameters to stack
            root_node.ir.variables.extend(self.params.iter().map(|p| {
                (*m, self.make_variable(NodeId(0)))
            }));
            // Add statics to stack
            root_node
                .ir
                .variables
                .extend(self.static_registers.iter().map(|r| {
                    var_cnt = var_cnt + 1;
                    (*r, VariableId(var_cnt))
                }))
                */
        }

        let mut made = HashSet::new();
        for i in 0..self.nodes.len() {
            self.propogate_variables(NodeId(i), vc, &mut made);
        }

        Ok(())
    }

    fn analyze_nodes(&mut self) -> Result<(), DecompileError> {
        // Get entry of each node
        let mut heads = HashSet::new();
        println!("Branches: {:?}", self.branches);
        println!("References: {:?}", self.references);
        for offset in 1..self.func.code.len() {
            if self.references[offset].len() > 1 {
                heads.insert(offset);
            }
            if self.branches[offset].len() > 1 {
                heads.extend(self.branches[offset].iter());
            }
        }

        // Create root node.
        {
            let mut root_node = Node::new(NodeId(0), 0);
            root_node.add_code(self, 0)?;
            self.nodes.push(RefCell::new(root_node));
        }

        // Create nodes
        for head in heads {
            println!("Making node at {}", head);
            let mut node = Node::new(NodeId(self.nodes.len()), head);
            node.add_code(self, head)?;
            self.nodes.push(RefCell::new(node));
        }

        // Add next & prev data
        for id in 0..self.nodes.len() {
            let last_offset = self.nodes[id].borrow().last_offset;
            for branch in &self.branches[last_offset] {
                let next = self.node_at(*branch).unwrap();
                self.nodes[id].borrow_mut().add_next(next);
                self.nodes[next.0].borrow_mut().add_prev(NodeId(id));
            }
        }
        Ok(())
    }

    fn analyze_branches(&mut self) -> Result<(), DecompileError> {
        let func = self.func.clone();
        let mut iter = func.code.iter().enumerate().peekable();
        while let Some((offset, i)) = iter.next() {
            match i.opcode()? {
                OpCode::Eq
                | OpCode::Lt
                | OpCode::Le
                | OpCode::TestSet
                | OpCode::Test
                | OpCode::TForLoop => {
                    let (_, next) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                    let target = (offset as i32 + next.argsbx() + 2) as usize;
                    if target >= self.func.code.len() {
                        return Err(DecompileError::UnexpectedEnd);
                    }
                    self.add_branch(offset, target)?;
                    self.add_branch(offset, offset + 2)?;
                }
                OpCode::ForPrep | OpCode::Jmp => {
                    self.add_branch(offset, (offset as i32 + i.argsbx() + 1) as usize)?;
                }
                OpCode::ForLoop => {
                    self.add_branch(offset, (offset as i32 + i.argsbx() + 1) as usize)?;
                    self.add_branch(offset, offset + 1)?;
                }
                OpCode::Closure => {
                    let nup = self.func.closures[i.argbx() as usize].nups as usize;
                    self.add_branch(offset, offset + nup + 1)?;
                    for _ in 0..nup {
                        iter.next();
                    }
                }
                OpCode::LoadBool => {
                    if i.argc() != 0 {
                        self.add_branch(offset, offset + 2)?;
                        iter.next();
                    } else {
                        self.add_branch(offset, offset + 1)?;
                    }
                }
                OpCode::SetList => {
                    // SETLIST can use the next instruction as a parameter
                    if i.argc() == 0 {
                        self.add_branch(offset, offset + 2)?;
                        iter.next();
                    } else {
                        self.add_branch(offset, offset + 1)?;
                    }
                }
                OpCode::TailCall | OpCode::Return => {
                    // No branches
                }
                _ => {
                    self.add_branch(offset, offset + 1)?;
                }
            }
        }

        Ok(())
    }

    /*
    fn print_value(&self, symbol: &Symbol) {
        if !symbol.label.is_empty() {
            print!("{}", symbol.label);
            return;
        }

        match &symbol.value {
            Value::Constant(c) => print!("{}", c),
            Value::Table { items } => {
                print!("{{");
                for (i, item) in items.iter().enumerate() {
                    let item = match item {
                        Some(i) => i,
                        None => panic!("Empty entry in table"),
                    };
                    if i != 0 {
                        print!(", ");
                    }
                    self.print_value(&item.borrow());
                }
                print!("}}");
            }
            Value::Global(c) => match c {
                Constant::Boolean(d) => print!("@{:X}", d),
                Constant::String(s) => print!("{}", s),
                _ => print!("_G[{}]", c),
            },
            Value::Return(c, true) => {
                let call = c.upgrade().unwrap();
                if let Value::Call {
                    func,
                    params,
                    returns,
                } = &call.borrow().value
                {
                    assert!(returns.len() == 1);
                    self.print_call(&func.borrow(), &params);
                };
            }
            _ => print!("unimplemented {:?}", symbol),
        }
    }

    fn print_call(&self, func: &Symbol, params: &[Symbol]) {
        self.print_value(func);
        print!("(");
        for (i, p) in params.iter().enumerate() {
            if i != 0 {
                print!(", ");
            }
            self.print_value(&p.borrow());
        }
        print!(")");
    }

    fn print_symbol(&self, symbol: &mut Symbol) {
        if !symbol.label.is_empty() {
            // print!("{}", symbol.label);
            return;
        }
        match symbol.value {
            Value::GetVarArgs(ref args) => {
                print!("local ");
                for (i, argr) in args.iter().enumerate() {
                    let mut arg = RefCell::borrow_mut(argr);
                    assert!(arg.label.is_empty());
                    arg.set_label(self.root.make_local());
                    if i != 0 {
                        print!(", ");
                    }
                    print!("{}", arg.label);
                }
                println!(" = ...");
            }
            Value::SetUpvalue(ref upval, ref src) => {
                {
                    let mut upval = RefCell::borrow_mut(upval);
                    if upval.label.is_empty() {
                        println!("-- FIXME: Upvalue should have been used before here");
                        upval.label = self.root.make_local();
                    }
                    print!("{} = ", upval.label);
                }
                self.print_value(&src.borrow());
                println!();
            }
            Value::Closure { index } => {
                let closure = &self.closures[index];
                symbol.set_label(self.root.make_local());
                print!("local {} = function(", symbol.label);
                for (i, p) in closure.params.iter().enumerate() {
                    let mut p = RefCell::borrow_mut(p);
                    p.set_label(self.root.make_local());
                    if i != 0 {
                        print!(", ");
                    }
                    print!("{}", p.label);
                }
                println!(")");
                closure.print();
                println!("end");
            }
            Value::Call {
                ref func,
                ref params,
                ref returns,
            } => {
                if !returns.is_empty() {
                    for (i, ret) in returns.iter().enumerate() {
                        let mut ret = RefCell::borrow_mut(ret);
                        if let Value::Return(_, va) = ret.value {
                            if va {
                                // Multiret; the next instruction should use this
                                println!("-- Multiret call declared here");
                                return;
                            }
                        }
                        if i != 0 {
                            print!(", ");
                        } else {
                            print!("local ");
                        }
                        assert!(ret.label.is_empty());
                        let label = self.root.make_local();
                        print!("{}", label);
                        ret.set_label(label);
                    }
                    print!(" = ");
                }
                self.print_call(&func.borrow(), &params);
                println!();
            }
            _ if symbol.must_define() => {
                let label = self.root.make_local();
                print!("local {} = ", label);
                self.print_value(&symbol);
                println!();
                symbol.set_label(label);
            }
            _ => {} //println!("unimplemented {:?}", symbol),
        }
    }


    fn print_node(&self, node: &Node, printed: &mut HashSet<usize>) {
        if !printed.insert(node.offset) {
            // Already printed
            return;
        }

        for symbol in &node.ir.symbols {
            //let symbol = symbol.borrow();
            self.print_symbol(&mut RefCell::borrow_mut(symbol));
        }

        match &node.tail {
            Tail::None => todo!(),
            Tail::Return(vals) => {
                print!("return ");
                for (i, val) in vals.iter().enumerate() {
                    if i != 0 {
                        print!(", ");
                    }
                    self.print_value(&RefCell::borrow(val));
                }
                println!();
            }
            Tail::TailCall(_) => todo!(),
            Tail::Eq(_) => todo!(),
            Tail::Le(_) => todo!(),
            Tail::Lt(_) => todo!(),
            Tail::TestSet(_, _) => todo!(),
            Tail::Test(_) => todo!(),
            Tail::TForLoop {
                call,
                index,
                state,
                inner,
                end,
            } => todo!(),
            Tail::ForLoop {
                init,
                limit,
                step,
                idx,
                inner,
                end,
            } => todo!(),
        };
    }

    fn print(&self) {
        let mut printed = HashSet::new();
        self.print_node(self.nodes.first().unwrap(), &mut printed);
    }*/

    fn print_node(&self, node: NodeId) {
        let node = &self.nodes[node.0].borrow();

        node.ir.print();

        match &node.ir.tail {
            Tail::None => {}
            Tail::Return(_) => todo!(),
            Tail::TailCall(_) => todo!(),
            Tail::Eq(cond) => {
                let node_1 = self.node_at(cond.target_1).unwrap();
                let node_2 = self.node_at(cond.target_1).unwrap();
                // print!("if {} {} {} then",
            }
            Tail::Le(_) => todo!(),
            Tail::Lt(_) => todo!(),
            Tail::TestSet(_, _) => todo!(),
            Tail::Test(_) => todo!(),
            Tail::TForLoop {
                call,
                index,
                state,
                inner,
                end,
            } => todo!(),
            Tail::ForLoop {
                init,
                limit,
                step,
                idx,
                inner,
                end,
            } => todo!(),
        }
    }

    fn print(&self) {
        self.print_node(NodeId(0));
    }

    fn decompile(&mut self) -> Result<(), DecompileError> {
        self.analyze_branches()?;
        self.analyze_nodes()?;
        let mut vc = VariableContext::new();
        self.make_variables(&mut vc)?;
        println!("{:#?}", vc);

        for closure in &mut self.closures {
            closure.decompile()?;
        }
        Ok(())
    }
}

pub fn decompile(root: Rc<RootContext>, func: Rc<Function>) -> Result<(), DecompileError> {
    // Generate
    let mut ctx = FunctionContext::new(root, func);
    ctx.decompile()?;

    //println!("{:#?}", ctx);

    ctx.print();

    //let ctx = FunctionContext { func };

    Ok(())
}
