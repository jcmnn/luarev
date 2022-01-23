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
}

pub enum Tail {
    None,
    Return,
    TailCall(Rc<DValue>, Vec<Rc<DValue>>),
    Eq {
        left: Rc<DValue>,
        right: Rc<DValue>,
        direciton: bool,
        target_1: usize,
        target_2: usize,
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

// Graph node
pub struct Node {
    offset: usize,
    ir: Vec<Rc<DValue>>,
    stack: NodeStack,
    tail: Tail,
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

    fn add_code(
        &mut self,
        ctx: &mut FunctionContext,
        code: &[Instruction],
    ) -> Result<(), DecompileError> {
        let mut iter = code.iter();
        while let Some(i) = iter.next() {
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
                        let upi = iter.next().ok_or(DecompileError::UnexpectedEnd)?;
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
                    let val = Rc::new(DValue::new(Value::Closure(bx)));
                    self.stack.set(i.arga() as usize, val.clone());
                    self.ir.push(val);
                }
                OpCode::Eq => {
                    
                }
                OpCode::Div => todo!(),
                OpCode::GetNum => todo!(),
                OpCode::Concat => todo!(),
                OpCode::GetTable => todo!(),
                OpCode::SetList => todo!(),
                OpCode::LoadK => todo!(),
                OpCode::SetGlobal => todo!(),
                OpCode::Jmp => todo!(),
                OpCode::TForLoop => todo!(),
                OpCode::SetUpval => todo!(),
                OpCode::Not => todo!(),
                OpCode::Vararg => todo!(),
                OpCode::GetUpval => todo!(),
                OpCode::Add => todo!(),
                OpCode::Return => todo!(),
                OpCode::GetGlobal => todo!(),
                OpCode::Len => todo!(),
                OpCode::Mul => todo!(),
                OpCode::NewTable => todo!(),
                OpCode::TestSet => todo!(),
                OpCode::SetTable => todo!(),
                OpCode::Unm => todo!(),
                OpCode::Mod => todo!(),
                OpCode::Lt => todo!(),
                OpCode::ForLoop => todo!(),
                OpCode::Call => todo!(),
                OpCode::Le => todo!(),
                OpCode::LoadBool => todo!(),
                OpCode::ForPrep => todo!(),
                OpCode::SetCGlobal => todo!(),
                OpCode::Test => todo!(),
                OpCode::Pow => todo!(),
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
