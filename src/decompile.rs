use std::{
    borrow::BorrowMut,
    cell::{Cell, RefCell},
    collections::HashSet,
    ops::Add,
    rc::{Rc, Weak},
};

use int_enum::IntEnumError;
use thiserror::Error;

use crate::{
    function::{Constant, Function, LvmInstruction, Name, OpCode},
    ir::{IrContext, StackId, Symbol, SymbolRef, Value},
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

#[derive(Debug)]
pub struct ConditionalA {
    value: SymbolRef,
    direction: bool,
    target_1: usize,
    target_2: usize,
}

#[derive(Debug)]
pub struct ConditionalB {
    left: SymbolRef,
    right: SymbolRef,
    direction: bool,
    target_1: usize,
    target_2: usize,
}

#[derive(Debug)]
pub enum Tail {
    None,
    Return(Vec<SymbolRef>),
    TailCall(SymbolRef),
    Eq(ConditionalB),
    Le(ConditionalB),
    Lt(ConditionalB),
    TestSet(ConditionalA, SymbolRef),
    Test(ConditionalA),
    TForLoop {
        call: SymbolRef,
        index: SymbolRef,
        state: SymbolRef,
        inner: usize,
        end: usize,
    },
    ForLoop {
        init: SymbolRef,
        limit: SymbolRef,
        step: SymbolRef,
        idx: SymbolRef,
        inner: usize,
        end: usize,
    },
}

// Graph node
#[derive(Debug)]
pub struct Node {
    offset: usize,
    last_offset: usize,
    ir: IrContext,
    tail: Tail,
    next: RefCell<Vec<Weak<Node>>>,
    prev: RefCell<Vec<Weak<Node>>>,
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
    fn new(offset: usize) -> Node {
        Node {
            offset,
            last_offset: offset,
            ir: IrContext::new(),
            tail: Tail::None,
            next: RefCell::new(Vec::new()),
            prev: RefCell::new(Vec::new()),
        }
    }

    fn ends_at(&self, end: &Rc<Node>) -> bool {
        false
    }

    fn add_next(&self, next: &Rc<Node>) {
        self.next.borrow_mut().push(Rc::downgrade(next));
    }

    fn add_prev(&self, prev: &Rc<Node>) {
        self.prev.borrow_mut().push(Rc::downgrade(prev));
    }

    // Returns value on stack or constant from encoded value
    fn stack_or_const(&mut self, r: u32, ctx: &FunctionContext) -> SymbolRef {
        if isk(r) {
            return self.ir.make_constant(ctx.func.constants[indexk(r)].clone());
        }

        self.ir.get_stack(StackId::from(r))
    }

    // Adds instructions to graph node
    fn add_code(
        &mut self,
        ctx: &mut FunctionContext,
        mut offset: usize,
    ) -> Result<(), DecompileError> {
        loop {
            let i = ctx.func.code[offset];
            // Decode conditional branch instruction
            let mut decode_conditionalb = || {
                let left = self.stack_or_const(i.argb(), ctx);
                let right = self.stack_or_const(i.argc(), ctx);

                let ijmp = ctx
                    .func
                    .code
                    .get(offset + 1)
                    .ok_or(DecompileError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(DecompileError::ExpectedJmp);
                }

                let target = offset as i32 + ijmp.argsbx() + 2;
                Ok(ConditionalB {
                    left,
                    right,
                    direction: i.arga() == 0,
                    target_1: offset + 2,
                    target_2: target as usize,
                })
            };

            // Decode instruction
            match i.opcode()? {
                OpCode::TailCall => {
                    let ra = i.arga() as i32;
                    let nparams = i.argb() as i32 - 1;
                    let nresults = i.argc() as i32 - 1;
                    if nresults != -1 {
                        println!("Tail call is not multiret. LVM will throw");
                        // not multiret; LVM throws here
                    }
                    let func = self.ir.get_stack(StackId::from(ra));
                    let call = self.ir.call(
                        func,
                        StackId::from(ra + 1),
                        nparams,
                        StackId::from(ra),
                        nresults,
                    );
                    self.tail = Tail::TailCall(call);
                }
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
                                OpCode::Move => Ok(self.ir.get_stack(StackId::from(upi.argb()))),
                                _ => Err(DecompileError::InvalidUpvalue),
                            }
                        })
                        .collect::<Result<Vec<SymbolRef>, DecompileError>>()?;

                    // Replace upvalues in closure
                    ctx.closures[bx].upvalues = upvalues; //.extend_from_slice(&upvalues);

                    self.ir
                        .closure(StackId::from(i.arga()), bx, &ctx.closures[bx].upvalues);
                }
                OpCode::Eq => {
                    self.tail = Tail::Eq(decode_conditionalb()?);
                }
                OpCode::Div => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.div(StackId::from(i.arga()), left, right);
                }
                OpCode::GetNum => {
                    self.ir.set_number(StackId::from(i.arga()), i.number());
                }
                OpCode::Concat => {
                    let b = i.argb();
                    let c = i.argc();

                    let params: Vec<SymbolRef> = (b..=c)
                        .map(|x| self.ir.get_stack(StackId::from(x)))
                        .collect();

                    self.ir.concat(StackId::from(i.arga()), params);
                }
                OpCode::GetTable => {
                    let table = self.ir.get_stack(StackId::from(i.argb()));
                    let key = self.stack_or_const(i.argc(), ctx);
                    self.ir.gettable(StackId::from(i.arga()), table, key);
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

                    let table = self.ir.get_stack(StackId::from(ra));
                    let mut table_ref = RefCell::borrow_mut(&table);
                    let items = match &mut table_ref.value {
                        Value::Table { items } => items,
                        _ => return Err(DecompileError::ExpectedTable),
                    };

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
                    }
                }
                OpCode::LoadK => {
                    let val = self
                        .ir
                        .make_constant(ctx.func.constants[i.argbx() as usize].clone());
                    self.ir.set_stack(StackId::from(i.arga()), val);
                }
                OpCode::SetGlobal => {
                    let cval = ctx.func.constants[i.argbx() as usize].clone();
                    self.ir.set_global(cval, StackId::from(i.arga()));
                }
                OpCode::Jmp => {}
                OpCode::TForLoop => {
                    let ra = i.arga() as usize;
                    let index = self.ir.get_stack(StackId::from(ra + 2));
                    let state = self.ir.get_stack(StackId::from(ra + 1));
                    let func = self.ir.get_stack(StackId::from(ra));
                    let nresults = i.argc() as i32;

                    //self.stack.set(ra + 3, func.clone());
                    //self.stack.set(ra + 4, state);
                    //self.stack.set(ra + 5, index);

                    let call = self.ir.call(
                        func,
                        StackId::from(ra + 1),
                        2,
                        StackId::from(ra + 3),
                        nresults,
                    );

                    let nexti = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    let target = ((offset as i32) + 2 + nexti.argsbx()) as usize;
                    self.tail = Tail::TForLoop {
                        call,
                        index,
                        state,
                        inner: target,
                        end: offset + 2,
                    };
                }
                OpCode::SetUpval => {
                    let src = self.ir.get_stack(StackId::from(i.arga()));
                    let upvalue = ctx.upvalues[i.argb() as usize].clone();
                    self.ir.add_symbol(Symbol::set_upvalue(upvalue, src));
                }
                OpCode::Not => {
                    let res = Symbol::not(self.ir.get_stack(StackId::from(i.argb())));
                    self.ir.set_stack_new(StackId::from(i.arga()), res);
                }
                OpCode::Vararg => {
                    let ra = i.arga();
                    let b = i.argb() as i32 - 1;
                    if b == -1 {
                        // Multret
                        self.ir
                            .set_stack_new(StackId::from(ra), Symbol::new(Value::VarArgs));
                    } else {
                        self.ir.get_varargs(StackId::from(ra), b as usize);
                    }
                }
                OpCode::GetUpval => {
                    self.ir.set_stack(
                        StackId::from(i.arga()),
                        ctx.upvalues[i.argb() as usize].clone(),
                    );
                }
                OpCode::Add => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir.add(StackId::from(i.arga()), left, right);
                }
                OpCode::Return => {
                    let ra = i.arga() as usize;
                    let nresults = i.argb() as i32 - 1;
                    let mut results = Vec::new();
                    if nresults == -1 {
                        // Add results until we find a variable type on the stack or reach the top
                        for j in ra..self.ir.stack.len() {
                            let val = self.ir.get_stack(StackId::from(j));
                            if val.borrow().is_var() {
                                results.push(val);
                                break;
                            } else {
                                results.push(val);
                            }
                        }
                    } else {
                        results.reserve(nresults as usize);
                        for j in 0..nresults {
                            results.push(self.ir.get_stack(StackId::from(ra as i32 + j)));
                        }
                    }
                    self.tail = Tail::Return(results);
                }
                OpCode::GetGlobal => {
                    let cval = ctx.func.constants[i.argbx() as usize].clone();
                    let global = Symbol::new(Value::Global(cval));
                    self.ir.set_stack_new(StackId::from(i.arga()), global);
                }
                OpCode::Len => {
                    let src = self.ir.get_stack(StackId::from(i.argb()));
                    let val = Symbol::len(src);
                    self.ir.set_stack_new(StackId::from(i.arga()), val);
                }
                OpCode::Mul => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = Symbol::mul(left, right);
                    self.ir.set_stack_new(StackId::from(i.arga()), val);
                }
                OpCode::NewTable => {
                    let table = Symbol::new(Value::Table { items: Vec::new() });
                    RefCell::borrow_mut(&table).force_define = true; // For debugging
                    self.ir.set_stack_new(StackId::from(i.arga()), table);
                }
                OpCode::TestSet => {
                    let test = self.ir.get_stack(StackId::from(i.argb()));
                    let original = self.ir.get_stack(StackId::from(i.arga()));
                    RefCell::borrow_mut(&original).add_reference(&test);

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = offset as i32 + ijmp.argsbx() + 2;
                    self.tail = Tail::TestSet(
                        ConditionalA {
                            value: test,
                            direction: i.arga() == 0,
                            target_1: offset + 2,
                            target_2: target as usize,
                        },
                        original,
                    );
                }
                OpCode::SetTable => {
                    let table = self.ir.get_stack(StackId::from(i.arga()));
                    let key = self.stack_or_const(i.argb(), ctx);
                    let value = self.stack_or_const(i.argc(), ctx);

                    self.ir.add_symbol(Symbol::settable(table, key, value));
                }
                OpCode::Unm => {
                    let src = self.ir.get_stack(StackId::from(i.argb()));
                    let val = Symbol::unm(src);
                    self.ir.set_stack_new(StackId::from(i.arga()), val);
                }
                OpCode::Mod => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = Symbol::modulus(left, right);
                    self.ir.set_stack_new(StackId::from(i.arga()), val);
                }
                OpCode::Lt => {
                    self.tail = Tail::Lt(decode_conditionalb()?);
                }
                OpCode::ForLoop => {
                    let ra = i.arga() as usize;
                    let step = self.ir.get_stack(StackId::from(ra + 2));
                    let limit = self.ir.get_stack(StackId::from(ra + 1));
                    let init = self.ir.get_stack(StackId::from(ra));

                    let idx = Symbol::new(Value::ForIndex);
                    let target = (offset as i32 + 1 + i.argsbx()) as usize;

                    self.tail = Tail::ForLoop {
                        init,
                        limit,
                        step,
                        idx,
                        inner: target,
                        end: offset + 1,
                    };
                }
                OpCode::Call => {
                    let ra = i.arga() as usize;
                    let func = self.ir.get_stack(StackId::from(ra));
                    let nparams = i.argb() as i32 - 1;
                    let nresults = i.argc() as i32 - 1;
                    self.ir.call(
                        func,
                        StackId::from(ra + 1),
                        nparams,
                        StackId::from(ra),
                        nresults,
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
                OpCode::Le => {
                    self.tail = Tail::Le(decode_conditionalb()?);
                }
                OpCode::LoadBool => {
                    let val = Symbol::boolean(i.argb() != 0);
                    self.ir.set_stack_new(StackId::from(i.arga()), val);
                }
                OpCode::ForPrep => {
                    // We don't need to do anything here
                }
                OpCode::SetCGlobal => {
                    let cval = ctx.func.constants[i.argbx() as usize].clone();
                    let src = self.ir.get_stack(StackId::from(i.arga()));
                    self.ir.add_symbol(Symbol::set_cglobal(cval, src));
                }
                OpCode::Test => {
                    let test = self.ir.get_stack(StackId::from(i.arga()));

                    let ijmp = ctx
                        .func
                        .code
                        .get(offset + 1)
                        .ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = offset as i32 + ijmp.argsbx() + 2;
                    self.tail = Tail::Test(ConditionalA {
                        value: test,
                        direction: i.arga() == 0,
                        target_1: offset + 2,
                        target_2: target as usize,
                    });
                }
                OpCode::Pow => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir
                        .set_stack_new(StackId::from(i.arga()), Symbol::pow(left, right));
                }
                OpCode::OpSelf => {
                    let table = self.ir.get_stack(StackId::from(i.argb()));
                    let key = self.stack_or_const(i.argc(), ctx);
                    self.ir
                        .set_stack(StackId::from(i.arga() + 1), table.clone());
                    self.ir.gettable(StackId::from(i.arga()), table, key);
                }
                OpCode::Sub => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = Symbol::sub(left, right);
                    self.ir.set_stack_new(StackId::from(i.arga()), val);
                }
                OpCode::Move => {
                    let val = self.ir.get_stack(StackId::from(i.argb()));
                    self.ir.set_stack(StackId::from(i.arga()), val);
                }
                OpCode::Close => {}
                OpCode::LoadNil => {
                    let ra = i.arga();
                    let rb = i.argb();
                    for ri in ra..=rb {
                        self.ir.set_stack_new(StackId::from(ri), Symbol::nil());
                    }
                }
            }

            if ctx.branches[offset].len() == 1 {
                let next_offset = ctx.branches[offset][0];
                if ctx.references[next_offset].len() > 1 {
                    break;
                }
                offset = next_offset;
                assert!(matches!(self.tail, Tail::None));
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

#[derive(Debug)]
pub struct FunctionContext {
    func: Rc<Function>,
    nodes: Vec<Rc<Node>>,
    branches: Vec<Vec<usize>>,
    params: Vec<SymbolRef>,
    references: Vec<Vec<usize>>,
    root: Rc<RootContext>,
    upvalues: Vec<SymbolRef>,
    closures: Vec<FunctionContext>,
}

impl FunctionContext {
    fn new(root: Rc<RootContext>, func: Rc<Function>) -> FunctionContext {
        let branches = vec![Vec::new(); func.code.len()];
        let references = vec![Vec::new(); func.code.len()];
        FunctionContext {
            upvalues: (0..func.nups).map(|_| Symbol::upvalue()).collect(),
            closures: func
                .closures
                .iter()
                .map(|f| FunctionContext::new(root.clone(), f.clone()))
                .collect(),
            params: (0..func.num_params)
                .map(|_| Symbol::new(Value::Param))
                .collect(),
            func,
            nodes: Vec::new(),
            branches,
            references,
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

    fn node_at(&self, offset: usize) -> Option<&Rc<Node>> {
        self.nodes.iter().find(|&x| x.offset == offset)
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
            let mut root_node = Rc::new(Node::new(0));
            {
                let node = Rc::get_mut(&mut root_node).unwrap();
                // Add function parameters to stack
                for (i, param) in self.params.iter().enumerate() {
                    node.ir.set_stack(StackId::from(i), param.clone());
                }
                node.add_code(self, 0)?;
            }
            self.nodes.push(root_node);
        }

        // Create nodes
        for head in heads {
            println!("Making node at {}", head);
            let mut node = Rc::new(Node::new(head));
            Rc::get_mut(&mut node).unwrap().add_code(self, head)?;
            self.nodes.push(node);
        }

        // Add next & prev data
        for node in &self.nodes {
            for branch in &self.branches[node.last_offset] {
                let next = self.node_at(*branch).unwrap();
                node.add_next(next);
                next.add_prev(node);
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
                    let target = next.argsbx() as usize + offset + 2;
                    if target >= self.func.code.len() {
                        return Err(DecompileError::UnexpectedEnd);
                    }
                    self.add_branch(offset, target)?;
                    self.add_branch(offset, offset + 2)?;
                }
                OpCode::ForPrep | OpCode::Jmp => {
                    self.add_branch(offset, offset + i.argsbx() as usize + 1)?;
                }
                OpCode::ForLoop => {
                    self.add_branch(offset, offset + i.argsbx() as usize + 1)?;
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

    fn print_call(&self, func: &Symbol, params: &[SymbolRef]) {
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
    }

    fn decompile(&mut self) -> Result<(), DecompileError> {
        self.analyze_branches()?;
        self.analyze_nodes()?;

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

    println!("{:#?}", ctx);

    ctx.print();

    //let ctx = FunctionContext { func };

    Ok(())
}
