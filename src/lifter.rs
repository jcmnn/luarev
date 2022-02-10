use int_enum::IntEnumError;
use thiserror::Error;

use crate::{function::{Function, LvmInstruction, OpCode}, ir::IrTree};

#[derive(Debug, Error)]
pub enum LifterError {
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


pub fn lift(func: &Function) -> Result<IrTree, LifterError> {
    let code_flow = CodeFlow::generate(func)?;

    
}


struct CodeFlow {
    forward: Vec<Vec<usize>>,
    reverse: Vec<Vec<usize>>,
}

impl CodeFlow {
    pub fn generate(func: &Function) -> Result<CodeFlow, LifterError> {
        let code = &func.code;
        let mut forward = Vec::with_capacity(code.len());
        forward.resize(code.len(), Vec::new());
        let mut flow = CodeFlow {
            reverse: forward.clone(),
            forward,
        };

        let mut iter = code.iter().enumerate();
        while let Some((offset, i)) = iter.next() {
            match i.opcode()? {
                OpCode::Eq
                | OpCode::Lt
                | OpCode::Le
                | OpCode::TestSet
                | OpCode::Test
                | OpCode::TForLoop => {
                    let (_, next) = iter.next().ok_or(LifterError::UnexpectedEnd)?;
                    let target = (offset as i32 + next.argsbx() + 2) as usize;
                    if target >= code.len() {
                        return Err(LifterError::UnexpectedEnd);
                    }
                    flow.add_branch(offset, target)?;
                    flow.add_branch(offset, offset + 2)?;
                }
                OpCode::ForPrep | OpCode::Jmp => {
                    flow.add_branch(offset, (offset as i32 + i.argsbx() + 1) as usize)?;
                }
                OpCode::ForLoop => {
                    flow.add_branch(offset, (offset as i32 + i.argsbx() + 1) as usize)?;
                    flow.add_branch(offset, offset + 1)?;
                }
                OpCode::Closure => {
                    let nup = func.closures[i.argbx() as usize].nups as usize;
                    flow.add_branch(offset, offset + nup + 1)?;
                    for _ in 0..nup {
                        iter.next();
                    }
                }
                OpCode::LoadBool => {
                    if i.argc() != 0 {
                        flow.add_branch(offset, offset + 2)?;
                        iter.next();
                    } else {
                        flow.add_branch(offset, offset + 1)?;
                    }
                }
                OpCode::SetList => {
                    // SETLIST can use the next instruction as a parameter
                    if i.argc() == 0 {
                        flow.add_branch(offset, offset + 2)?;
                        iter.next();
                    } else {
                        flow.add_branch(offset, offset + 1)?;
                    }
                }
                OpCode::TailCall | OpCode::Return => {
                    // No branches
                }
                _ => {
                    flow.add_branch(offset, offset + 1)?;
                }
            }
        }
        Ok(flow)
    }

    fn add_branch(&mut self, src: usize, dst: usize) -> Result<(), LifterError> {
        if src >= self.forward.len() || dst >= self.reverse.len() {
            Err(LifterError::UnexpectedEnd)
        } else {
            self.forward[src].push(dst);
            self.reverse[dst].push(src);
            Ok(())
        }
    }
}