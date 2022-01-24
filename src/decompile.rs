use std::{
    cell::{Cell, RefCell},
    ops::Add,
    rc::Rc,
};

use int_enum::IntEnumError;
use thiserror::Error;

use crate::function::{DValue, Function, LvmInstruction, Name, OpCode, Value};

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

pub struct ConditionalA {
    value: Rc<DValue>,
    direction: bool,
    target_1: usize,
    target_2: usize,
}

pub struct ConditionalB {
    left: Rc<DValue>,
    right: Rc<DValue>,
    direction: bool,
    target_1: usize,
    target_2: usize,
}

pub enum Tail {
    None,
    Return(Vec<Rc<DValue>>),
    TailCall(Rc<DValue>, Vec<Rc<DValue>>),
    Eq(ConditionalB),
    Le(ConditionalB),
    Lt(ConditionalB),
    TestSet(ConditionalA),
    TForLoop {
        call: Rc<DValue>,
        inner: usize,
        end: usize,
    },
    ForLoop {
        init: Rc<DValue>,
        limit: Rc<DValue>,
        step: Rc<DValue>,
        idx: Rc<DValue>,
        inner: usize,
        end: usize,
    },
}

pub struct NodeStack {
    stack: Vec<Option<Rc<DValue>>>,
}

impl NodeStack {
    fn new() -> NodeStack {
        NodeStack { stack: Vec::new() }
    }

    fn get(&mut self, idx: usize) -> Rc<DValue> {
        if idx >= self.stack.len() {
            self.stack.resize(idx + 1, None);
        }
        let val = self.stack[idx]
            .get_or_insert_with(|| Rc::new(DValue::new(Value::Unknown(idx))))
            .clone();
        val.refcount.set(val.refcount.get() + 1);
        val
    }

    fn set(&mut self, idx: usize, val: Rc<DValue>) {
        if idx >= self.stack.len() {
            self.stack.resize(idx + 1, None);
        }
        self.stack[idx] = Some(val);
    }
}

pub struct NodeContext {

}

// Graph node
pub struct Node {
    offset: usize,
    ir: Vec<Rc<DValue>>,
    stack: NodeStack,
    tail: Tail,
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
            ir: Vec::new(),
            stack: NodeStack::new(),
            tail: Tail::None,
        }
    }

    // Returns value on stack or constant from encoded value
    fn stack_or_const(&mut self, r: u32, ctx: &FunctionContext) -> Rc<DValue> {
        if isk(r) {
            return self.make_value(Value::Constant(ctx.func.constants[indexk(r)].clone()));
        }

        self.stack.get(r as usize).clone()
    }

    fn make_value(&mut self, value: Value) -> Rc<DValue> {
        let dv = Rc::new(DValue::new(value));
        self.ir.push(dv.clone());
        dv
    }

    // Adds instructions to graph node
    fn add_code(
        &mut self,
        ctx: &mut FunctionContext,
        offset: usize,
        code: &[LvmInstruction],
    ) -> Result<(), DecompileError> {
        let mut iter = code.iter().enumerate().map(|(coff, i)| (coff + offset, i));
        while let Some((coff, i)) = iter.next() {
            // Decode conditional branch instruction
            let mut decode_conditionalb = || {
                let left = self.stack_or_const(i.argb(), ctx);
                let right = self.stack_or_const(i.argc(), ctx);

                let (_, ijmp) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(DecompileError::ExpectedJmp);
                }

                let target = coff as i32 + ijmp.argsbx() + 2;
                Ok(ConditionalB {
                    left,
                    right,
                    direction: i.arga() == 0,
                    target_1: coff + 2,
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
                        // not multiret; LVM throws here
                    }
                    let func = self.stack.get(ra as usize);
                    let params: Vec<Rc<DValue>> = (0..nparams)
                        .map(|j| self.stack.get((ra + 1 + j) as usize))
                        .collect();
                    self.tail = Tail::TailCall(func, params);
                    assert!(iter.next().is_none());
                }
                OpCode::Closure => {
                    let bx = i.argbx() as usize;
                    if bx >= ctx.func.closures.len() {
                        println!("Invalid closure index");
                        continue;
                    }
                    let closure = &ctx.func.closures[bx];
                    for up in closure.upvalues.iter() {
                        let (_, upi) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                        let val = match upi.opcode()? {
                            OpCode::GetUpval => ctx.func.upvalues[upi.argb() as usize].clone(),
                            OpCode::Move => self.stack.get(upi.argb() as usize),
                            _ => return Err(DecompileError::InvalidUpvalue),
                        };
                        // Force name
                        if matches!(*val.name.borrow(), Name::None) {
                            val.name.replace(ctx.root.make_local());
                        }
                        up.name.replace((*val.name.borrow()).clone());
                    }
                    let val = self.make_value(Value::Closure(bx));
                    self.stack.set(i.arga() as usize, val.clone());
                    self.ir.push(val);
                }
                OpCode::Eq => {
                    self.tail = Tail::Eq(decode_conditionalb()?);
                    assert!(iter.next().is_none());
                }
                OpCode::Div => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Div(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::GetNum => {
                    let val = self.make_value(Value::Number(i.number()));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Concat => {
                    let b = i.argb();
                    let c = i.argc();

                    let params: Vec<Rc<DValue>> =
                        (b..=c).map(|x| self.stack.get(x as usize)).collect();

                    let val = self.make_value(Value::Concat(params));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::GetTable => {
                    let table = self.stack.get(i.argb() as usize);
                    let key = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::TableValue(table, key));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::SetList => {
                    let n = i.argb();
                    let c = match i.argc() {
                        0 => {
                            let (_, nexti) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                            nexti.raw()
                        }
                        c => c,
                    };

                    let ra = self.stack.get(i.arga() as usize);
                    let mut items = match &ra.value {
                        Value::NewTable(items) => items.borrow_mut(),
                        _ => return Err(DecompileError::ExpectedTable),
                    };
                    let mut last = (((c - 1) * 50) + n) as usize;

                    if n == 0 {
                        // Vararg
                        if last >= items.len() {
                            items.resize(last + 1, None);
                        }
                        items[last] = Some(Rc::new(DValue::new(Value::VarArg)));
                        continue;
                    }

                    // Resize table if needed
                    if last > items.len() {
                        items.resize(last + 1, None);
                    }

                    for idx in n..0 {
                        let val = self.stack.get((i.arga() + idx) as usize);
                        items[last] = Some(val);
                        last -= 1;
                    }
                }
                OpCode::LoadK => {
                    let val = self.make_value(Value::Constant(
                        ctx.func.constants[i.argbx() as usize].clone(),
                    ));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::SetGlobal => {
                    let cval = ctx.func.constants[i.argbx() as usize].clone();
                    let src = self.stack.get(i.arga() as usize);
                    self.make_value(Value::SetGlobal(cval, src));
                }
                OpCode::Jmp => {}
                OpCode::TForLoop => {
                    let ra = i.arga() as usize;
                    let _index = self.stack.get(ra + 2);
                    let _state = self.stack.get(ra + 1);
                    let func = self.stack.get(ra);
                    let nresults = i.argc() as usize;

                    //self.stack.set(ra + 3, func.clone());
                    //self.stack.set(ra + 4, state);
                    //self.stack.set(ra + 5, index);

                    let mut return_values = Vec::with_capacity(nresults);

                    for ri in 0..nresults {
                        let val = self.make_value(Value::ReturnValue(func.clone()));
                        return_values.push(val.clone());
                        self.stack.set(ra + 3 + ri, val);
                    }

                    let call = self.make_value(Value::Call(func, Vec::new(), return_values));

                    let (_, nexti) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                    let target = ((coff as i32) + 2 + nexti.argsbx()) as usize;
                    self.tail = Tail::TForLoop {
                        call,
                        inner: target,
                        end: coff + 2,
                    };
                    assert!(iter.next().is_none());
                }
                OpCode::SetUpval => {
                    let src = self.stack.get(i.arga() as usize);
                    self.make_value(Value::SetUpValue(i.argb() as usize, src));
                }
                OpCode::Not => {
                    let src = self.stack.get(i.argb() as usize);
                    let val = self.make_value(Value::Not(src));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Vararg => {}
                OpCode::GetUpval => {
                    let val = ctx.func.upvalues[i.argb() as usize].clone();
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Add => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Add(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Return => {
                    let ra = i.arga() as usize;
                    let nresults = i.argb() as i32 - 1;
                    let mut results = Vec::new();
                    if nresults == -1 {
                        let va = self.make_value(Value::VarArg);
                        results.push(va);
                    } else {
                        results.reserve(nresults as usize);
                        for j in 0..nresults {
                            results.push(self.stack.get(ra + j as usize));
                        }
                    }
                    self.tail = Tail::Return(results);
                    assert!(iter.next().is_none());
                }
                OpCode::GetGlobal => {
                    let cval = ctx.func.constants[i.argbx() as usize].clone();
                    let val = self.make_value(Value::GetGlobal(cval));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Len => {
                    let src = self.stack.get(i.argb() as usize);
                    let val = self.make_value(Value::Len(src));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Mul => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Mul(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::NewTable => {
                    let table = self.make_value(Value::NewTable(RefCell::new(Vec::new())));
                    self.stack.set(i.arga() as usize, table);
                }
                OpCode::TestSet => {
                    let test = self.stack.get(i.argb() as usize);
                    let _original = self.stack.get(i.arga() as usize);

                    let (_, ijmp) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = coff as i32 + ijmp.argsbx() + 2;
                    self.tail = Tail::TestSet(ConditionalA {
                        value: test,
                        direction: i.arga() == 0,
                        target_1: coff + 2,
                        target_2: target as usize,
                    });
                }
                OpCode::SetTable => {
                    let table = self.stack.get(i.arga() as usize);
                    let key = self.stack_or_const(i.argb(), ctx);
                    let value = self.stack_or_const(i.argc(), ctx);

                    let _dst = self.make_value(Value::SetTable(table, key, value));
                }
                OpCode::Unm => {
                    let src = self.stack.get(i.argb() as usize);
                    let val = self.make_value(Value::Unm(src));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Mod => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Mod(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Lt => {
                    self.tail = Tail::Lt(decode_conditionalb()?);
                    assert!(iter.next().is_none());
                }
                OpCode::ForLoop => {
                    let ra = i.arga() as usize;
                    let step = self.stack.get(ra + 2);
                    let limit = self.stack.get(ra + 1);
                    let init = self.stack.get(ra);

                    let idx = self.make_value(Value::ForIndex);
                    let target = (coff as i32 + 1 + i.argsbx()) as usize;

                    self.tail = Tail::ForLoop {
                        init,
                        limit,
                        step,
                        idx,
                        inner: target,
                        end: coff + 1,
                    };
                    assert!(iter.next().is_none());
                }
                OpCode::Call => {
                    let ra = i.arga() as usize;
                    let func = self.stack.get(ra);
                    let nparams = i.argb() as i32 - 1;
                    let nresults = i.argc() as i32 - 1;
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
                }
                OpCode::Le => {
                    self.tail = Tail::Le(decode_conditionalb()?);
                    assert!(iter.next().is_none());
                }
                OpCode::LoadBool => {
                    let val = self.make_value(Value::Boolean(i.argb() != 0));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::ForPrep => {
                    // We don't need to do anything here
                }
                OpCode::SetCGlobal => todo!(),
                OpCode::Test => todo!(),
                OpCode::Pow => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Pow(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::OpSelf => todo!(),
                OpCode::Sub => todo!(),
                OpCode::Move => todo!(),
                OpCode::Close => todo!(),
                OpCode::LoadNil => todo!(),
            }
        }
        Ok(())
    }
}

pub struct RootContext {
    local_idx: Cell<usize>,
}

impl RootContext {
    pub fn new() -> RootContext {
        RootContext {
            local_idx: Cell::new(0),
        }
    }

    pub fn make_local(&self) -> Name {
        let idx = self.local_idx.get();
        self.local_idx.set(idx + 1);
        Name::Local(format!("local_{}", idx))
    }
}

pub struct FunctionContext {
    func: Rc<Function>,
    nodes: Vec<Rc<Node>>,
    branches: Vec<Vec<usize>>,
    references: Vec<Vec<usize>>,
    root: Rc<RootContext>,
}

impl FunctionContext {
    fn new(root: Rc<RootContext>, func: Rc<Function>) -> FunctionContext {
        let branches = vec![Vec::new(); func.code.len()];
        let references = vec![Vec::new(); func.code.len()];
        FunctionContext {
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
                }
                OpCode::LoadBool => {
                    if i.argc() != 0 {
                        self.add_branch(offset, offset + 2)?;
                    } else {
                        self.add_branch(offset, offset + 1)?;
                    }
                }
                OpCode::SetList => {
                    // SETLIST can use the next instruction as a parameter
                    if i.argc() == 0 {
                        self.add_branch(offset, offset + 2)?;
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
}

pub fn decompile(root: Rc<RootContext>, func: Rc<Function>) -> Result<(), DecompileError> {
    // Generate
    let mut ctx = FunctionContext::new(root, func);
    ctx.analyze_branches()?;

    //let ctx = FunctionContext { func };

    Ok(())
}
