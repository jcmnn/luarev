use std::{cell::Cell, ops::Add, rc::Rc};

use int_enum::IntEnumError;
use thiserror::Error;

use crate::function::{DValue, Function, Instruction, Name, OpCode, Value};

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
}

pub struct Branch {
    left: Rc<DValue>,
    right: Rc<DValue>,
    direction: bool,
    target_1: usize,
    target_2: usize,
}

pub enum Tail {
    None,
    Return,
    TailCall(Rc<DValue>, Vec<Rc<DValue>>),
    Eq(Branch),
    Le(Branch),
    Lt(Branch),
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
        code: &[Instruction],
    ) -> Result<(), DecompileError> {
        let mut iter = code.iter().enumerate().map(|(coff, i)| (coff + offset, i));
        while let Some((coff, i)) = iter.next() {
            // Decode conditional branch instruction
            let mut decode_conditional = || {
                let left = self.stack_or_const(i.argb(), ctx);
                let right = self.stack_or_const(i.argc(), ctx);

                let (_, ijmp) = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(DecompileError::ExpectedJmp);
                }

                let target = coff as i32 + ijmp.argsbx() + 2;
                Ok(Branch {
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
                    self.tail = Tail::Eq(decode_conditional()?);
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
                    let b = i.argc();

                    todo!()
                }
                OpCode::LoadK => todo!(),
                OpCode::SetGlobal => todo!(),
                OpCode::Jmp => todo!(),
                OpCode::TForLoop => todo!(),
                OpCode::SetUpval => todo!(),
                OpCode::Not => todo!(),
                OpCode::Vararg => todo!(),
                OpCode::GetUpval => todo!(),
                OpCode::Add => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Add(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Return => todo!(),
                OpCode::GetGlobal => todo!(),
                OpCode::Len => todo!(),
                OpCode::Mul => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Mul(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::NewTable => todo!(),
                OpCode::TestSet => todo!(),
                OpCode::SetTable => todo!(),
                OpCode::Unm => todo!(),
                OpCode::Mod => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = self.make_value(Value::Mod(left, right));
                    self.stack.set(i.arga() as usize, val);
                }
                OpCode::Lt => {
                    self.tail = Tail::Lt(decode_conditional()?);
                }
                OpCode::ForLoop => todo!(),
                OpCode::Call => todo!(),
                OpCode::Le => {
                    self.tail = Tail::Le(decode_conditional()?);
                }
                OpCode::LoadBool => todo!(),
                OpCode::ForPrep => todo!(),
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
