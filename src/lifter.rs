use std::collections::HashSet;

use int_enum::IntEnumError;
use thiserror::Error;

use crate::{
    function::{Function, LvmInstruction, OpCode},
    ir::{
        ConditionalB, ConstantId, IrNodeBuilder, IrTree, RegConst, StackId, Tail, UpvalueId, Value,
        VariableSolver,
    },
};

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

// Returns true if encoded value contains a constant index
const fn isk(x: u32) -> bool {
    (x & (1 << 8)) != 0
}

// Returns constant index from encoded value
const fn indexk(x: u32) -> usize {
    (x & !(1 << 8)) as usize
}

// Returns value on stack or constant from encoded value
fn stack_or_const(r: u32) -> RegConst {
    if isk(r) {
        return RegConst::Constant(ConstantId(indexk(r)));
    }

    RegConst::Stack(StackId::from(r))
}

fn lift_node(
    func: &Function,
    head: usize,
    flow: &CodeFlow,
    solver: &mut VariableSolver,
    tree: &mut IrTree,
) -> Result<(), LifterError> {
    let mut node = IrNodeBuilder::new(solver);
    let mut offset = head;

    loop {
        let i = func.code[offset];

        // Decode instruction
        match i.opcode()? {
            OpCode::Closure => {
                let bx = i.argbx() as usize;
                if bx >= func.closures.len() {
                    println!("Invalid closure index");
                    break;
                }
                let closure = &func.closures[bx];
                let upvalues = (0..closure.nups)
                    .map(|idx| {
                        let upi = func
                            .code
                            .get(offset + 1 + idx as usize)
                            .ok_or(LifterError::UnexpectedEnd)?;
                        match upi.opcode()? {
                            OpCode::GetUpval => {
                                Ok(RegConst::UpValue(UpvalueId::from(upi.argb() as usize)))
                            }
                            OpCode::Move => {
                                let reg = StackId::from(upi.argb());
                                // Set register as static
                                tree.add_static(reg);
                                Ok(RegConst::Stack(reg))
                            }
                            _ => Err(LifterError::InvalidUpvalue),
                        }
                    })
                    .collect::<Result<Vec<RegConst>, LifterError>>()?;

                // Replace upvalues in closure TODO
                // ctx.closures[bx].upvalues = upvalues; //.extend_from_slice(&upvalues);

                node.closure(
                    StackId::from(i.arga()),
                    bx,
                    upvalues, //ctx.closures[bx].upvalues.clone(),
                );
            }
            OpCode::Div => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());
                node.div(StackId::from(i.arga()), left, right);
            }
            OpCode::GetNum => {
                node.number(StackId::from(i.arga()), i.number());
            }
            OpCode::Concat => {
                let b = i.argb();
                let c = i.argc();

                let params: Vec<RegConst> =
                    (b..=c).map(|x| RegConst::Stack(StackId::from(x))).collect();

                node.concat(StackId::from(i.arga()), params);
            }
            OpCode::GetTable => {
                let table = RegConst::Stack(StackId::from(i.argb()));
                let key = stack_or_const(i.argc());
                node.get_table(StackId::from(i.arga()), table, key);
            }
            OpCode::SetList => {
                let n = i.argb() as usize;
                let ra = i.arga() as usize;
                let c = match i.argc() {
                    0 => {
                        let nexti = func
                            .code
                            .get(offset + 1)
                            .ok_or(LifterError::UnexpectedEnd)?;
                        nexti.raw() as usize
                    }
                    c => c as usize,
                };
                assert!(c != 0);

                let table = StackId::from(ra);
                node.set_list(table, (c - 1) * 50, n);
            }
            OpCode::LoadK => {
                node.load_constant(
                    StackId::from(i.arga()),
                    ConstantId(i.argbx() as usize),
                    // func.constants[i.argbx() as usize].clone(),
                );
            }
            OpCode::SetGlobal => {
                let cval = ConstantId(i.argbx() as usize); // func.constants[i.argbx() as usize].clone();
                node.set_global(cval, RegConst::Stack(StackId::from(i.arga())));
            }
            OpCode::SetCGlobal => {
                let key = ConstantId(i.argbx() as usize); // func.constants[i.argbx() as usize].clone();
                node.set_cglobal(key, RegConst::Stack(StackId::from(i.arga())));
            }
            OpCode::Jmp => {}
            OpCode::TForLoop => {
                let ra = StackId::from(i.arga());
                let nresults = i.argc() as i32;

                let nexti = func
                    .code
                    .get(offset + 1)
                    .ok_or(LifterError::UnexpectedEnd)?;
                let target = ((offset as i32) + 2 + nexti.argsbx()) as usize;

                node.tail_tforloop(
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
                node.set_upvalue(UpvalueId::from(i.argb() as usize), src);
            }
            OpCode::Not => {
                node.not(
                    StackId::from(i.arga()),
                    RegConst::Stack(StackId::from(i.argb())),
                );
            }
            OpCode::Vararg => {
                let ra = i.arga();
                let b = i.argb() as i32 - 1;
                node.get_varargs(
                    StackId::from(ra),
                    if b == -1 { None } else { Some(b as usize) },
                )
            }
            OpCode::GetUpval => {
                node.get_upvalue(StackId::from(i.arga()), UpvalueId::from(i.argb() as usize));
            }
            OpCode::Add => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());
                node.add(StackId::from(i.arga()), left, right);
            }
            OpCode::Return => {
                let ra = StackId::from(i.arga());
                let nresults = i.argb() as i32 - 1;
                node.tail_return(
                    ra,
                    if nresults == -1 {
                        None
                    } else {
                        Some(nresults as usize)
                    },
                );
            }
            OpCode::GetGlobal => {
                let key = ConstantId(i.argbx() as usize); // func.constants[i.argbx() as usize].clone();
                node.get_global(StackId::from(i.arga()), key);
            }
            OpCode::Len => {
                node.len(
                    StackId::from(i.arga()),
                    RegConst::Stack(StackId::from(i.argb())),
                );
            }
            OpCode::Mul => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());
                node.mul(StackId::from(i.arga()), left, right);
            }
            OpCode::NewTable => {
                node.new_table(StackId::from(i.arga()));
            }
            OpCode::SetTable => {
                let table = RegConst::Stack(StackId::from(i.arga()));
                let key = stack_or_const(i.argb());
                let value = stack_or_const(i.argc());
                node.set_table(table, key, value);
            }
            OpCode::Unm => {
                node.unm(
                    StackId::from(i.arga()),
                    RegConst::Stack(StackId::from(i.argb())),
                );
            }
            OpCode::Mod => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());
                node.modulus(StackId::from(i.arga()), left, right);
            }
            OpCode::ForLoop => {
                let ra = StackId::from(i.arga());
                let step = RegConst::Stack(ra + 2_usize);
                let limit = RegConst::Stack(ra + 1_usize);
                let init = RegConst::Stack(ra);

                let target = (offset as i32 + 1 + i.argsbx()) as usize;
                node.tail_forloop(step, limit, init, ra + 3_usize, target, offset + 1)
            }
            OpCode::Call => {
                let ra = i.arga() as usize;
                let func = RegConst::Stack(StackId::from(ra));
                let nparams = i.argb() as i32 - 1;
                let nresults = i.argc() as i32 - 1;
                node.call(
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
            }
            OpCode::LoadBool => {
                node.load_boolean(StackId::from(i.arga()), i.argb() != 0);
            }
            OpCode::ForPrep => {
                // We don't need to do anything here
            }
            OpCode::Pow => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());
                node.pow(StackId::from(i.arga()), left, right);
            }
            OpCode::OpSelf => {
                // TODO: Dedicated self operation
                let table = RegConst::Stack(StackId::from(i.argb()));
                let key = stack_or_const(i.argc());
                let symbol = Value::Symbol(node.reference_regconst(table));
                node.set_stack(StackId::from(i.arga() + 1), symbol, true);
                node.get_table(StackId::from(i.arga()), table, key);
            }
            OpCode::Sub => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());
                node.sub(StackId::from(i.arga()), left, right);
            }
            OpCode::Move => {
                node.mov(StackId::from(i.arga()), StackId::from(i.argb()));
            }
            OpCode::Close => {}
            OpCode::LoadNil => {
                let ra = i.arga();
                let rb = i.argb();
                for ri in ra..=rb {
                    node.load_nil(StackId::from(ri));
                }
            }
            // Tails
            OpCode::Eq => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());

                let ijmp = func
                    .code
                    .get(offset + 1)
                    .ok_or(LifterError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(LifterError::ExpectedJmp);
                }

                let target = offset as i32 + ijmp.argsbx() + 2;
                node.tail_eq(left, right, i.arga() == 0, offset + 2, target as usize);
            }
            OpCode::Lt => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());

                let ijmp = func
                    .code
                    .get(offset + 1)
                    .ok_or(LifterError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(LifterError::ExpectedJmp);
                }

                let target = offset as i32 + ijmp.argsbx() + 2;
                node.tail_lt(left, right, i.arga() == 0, offset + 2, target as usize);
            }
            OpCode::Le => {
                let left = stack_or_const(i.argb());
                let right = stack_or_const(i.argc());

                let ijmp = func
                    .code
                    .get(offset + 1)
                    .ok_or(LifterError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(LifterError::ExpectedJmp);
                }

                let target = offset as i32 + ijmp.argsbx() + 2;
                node.tail_le(left, right, i.arga() == 0, offset + 2, target as usize);
            }
            OpCode::TestSet => {
                let test = RegConst::Stack(StackId::from(i.argb()));
                let dst = StackId::from(i.arga());

                let ijmp = func
                    .code
                    .get(offset + 1)
                    .ok_or(LifterError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(LifterError::ExpectedJmp);
                }

                node.tail_testset(
                    test,
                    dst,
                    i.arga() == 0,
                    offset + 2,
                    (offset as i32 + ijmp.argsbx() + 2) as usize,
                );
            }
            OpCode::Test => {
                let test = RegConst::Stack(StackId::from(i.arga()));

                let ijmp = func
                    .code
                    .get(offset + 1)
                    .ok_or(LifterError::UnexpectedEnd)?;
                if ijmp.opcode()? != OpCode::Jmp {
                    return Err(LifterError::ExpectedJmp);
                }

                node.tail_test(
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
                node.tail_tailcall(
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

        if flow.forward[offset].len() == 1 {
            let next_offset = flow.forward[offset][0];
            if flow.reverse[next_offset].len() > 1 {
                break;
            }
            offset = next_offset;
            assert!(matches!(node.tail, Tail::None));
        } else {
            break;
        }
    }

    match &node.tail {
        Tail::None => tree.connect_node(head, *flow.forward[offset].first().unwrap()),
        Tail::Return(_) => {}
        Tail::TailCall(_) => {}
        Tail::Eq(cond) | Tail::Le(cond) | Tail::Lt(cond) => {
            tree.connect_node(head, cond.target_1);
            tree.connect_node(head, cond.target_2);
        }
        Tail::TestSet(cond, _) | Tail::Test(cond) => {
            tree.connect_node(head, cond.target_1);
            tree.connect_node(head, cond.target_2);
        }
        Tail::TForLoop {
            call: _,
            index: _,
            state: _,
            inner,
            end,
        }
        | Tail::ForLoop {
            init: _,
            limit: _,
            step: _,
            idx: _,
            inner,
            end,
        } => {
            tree.connect_node(head, *inner);
            tree.connect_node(head, *end);
        }
    }

    tree.add_node(head, node.build());
    Ok(())
}

pub fn lift<'a, 'b>(
    func: &'a Function,
    solver: &'b mut VariableSolver,
) -> Result<IrTree<'a>, LifterError> {
    let code_flow = CodeFlow::generate(func)?;
    let node_heads = code_flow.nodes();

    let mut tree = IrTree::new(func);

    {
        // Root node with upvalues and params
        let mut builder = IrNodeBuilder::new(solver);
        for i in 0..func.num_params {
            builder.modify_stack(StackId::from(i));
        }

        tree.add_node(usize::MAX, builder.build());
    }

    for head in node_heads {
        lift_node(func, head, &code_flow, solver, &mut tree)?;
    }

    // Connect root to first node
    tree.connect_node(usize::MAX, 0);

    for closure in &func.closures {
        tree.closures.push(lift(&closure, solver)?);
    }

    Ok(tree)
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

        flow.add_branch(usize::MAX, 0)?;

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
        if (src != usize::MAX && src >= self.forward.len()) || dst >= self.reverse.len() {
            Err(LifterError::UnexpectedEnd)
        } else {
            if src != usize::MAX {
                self.forward[src].push(dst);
            }
            self.reverse[dst].push(src);
            Ok(())
        }
    }

    // Returns the entry point of nodes
    fn nodes(&self) -> Vec<usize> {
        let mut heads = HashSet::new();
        // Add root node
        heads.insert(0);

        for offset in 1..self.forward.len() {
            if self.reverse[offset].len() > 1 {
                heads.insert(offset);
            }
            if self.forward[offset].len() > 1 {
                heads.extend(self.forward[offset].iter());
            }
        }
        Vec::from_iter(heads)
    }
}
