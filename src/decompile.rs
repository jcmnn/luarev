use std::{
    cell::{Cell, RefCell},
    ops::Add,
    rc::Rc, borrow::BorrowMut, collections::HashSet,
};

use int_enum::IntEnumError;
use thiserror::Error;

use crate::{
    function::{Function, LvmInstruction, Name, OpCode},
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
    ir: IrContext,
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
            ir: IrContext::new(),
            tail: Tail::None,
        }
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

                let ijmp = ctx.func.code.get(offset + 1).ok_or(DecompileError::UnexpectedEnd)?;
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
                            let upi = ctx.func.code.get(offset + idx as usize).ok_or(DecompileError::UnexpectedEnd)?;
                            match upi.opcode()? {
                                OpCode::GetUpval => Ok(ctx.upvalues[upi.argb() as usize].clone()),
                                OpCode::Move => Ok(self.ir.get_stack(StackId::from(upi.argb()))),
                                _ => Err(DecompileError::InvalidUpvalue),
                            }
                        })
                        .collect::<Result<Vec<SymbolRef>, DecompileError>>()?;

                    self.ir.closure(StackId::from(i.arga()), bx, upvalues);
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
                    let n = i.argb();
                    let c = match i.argc() {
                        0 => {
                            let nexti = ctx.func.code.get(offset + 1).ok_or(DecompileError::UnexpectedEnd)?;
                            nexti.raw()
                        }
                        c => c,
                    };

                    let table = self.ir.get_stack(StackId::from(i.arga()));
                    let mut table_ref = RefCell::borrow_mut(&table);
                    let items = match &mut table_ref.value {
                        Value::Table { items } => items,
                        _ => return Err(DecompileError::ExpectedTable),
                    };
                    let mut last = (((c - 1) * 50) + n) as usize;

                    if n == 0 {
                        // Vararg
                        if last >= items.len() {
                            items.resize(last + 1, None);
                        }
                        let val = self.ir.add_symbol(Symbol::new(Value::VarArg));
                        RefCell::borrow_mut(&val).add_reference(&table);
                        items[last] = Some(val);
                        break;
                    }

                    // Resize table if needed
                    if last > items.len() {
                        items.resize(last + 1, None);
                    }

                    for idx in n..0 {
                        let val = self.ir.get_stack(StackId::from(i.arga() + idx));
                        RefCell::borrow_mut(&val).add_reference(&table);
                        items[last] = Some(val);
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
                    let _index = self.ir.get_stack(StackId::from(ra + 2));
                    let _state = self.ir.get_stack(StackId::from(ra + 1));
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

                    offset += 1;
                    let nexti = ctx.func.code.get(offset).ok_or(DecompileError::UnexpectedEnd)?;
                    let target = ((offset as i32) + 2 + nexti.argsbx()) as usize;
                    self.tail = Tail::TForLoop {
                        call,
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
                    self.ir.set_stack(StackId::from(i.arga()), res);
                }
                OpCode::Vararg => {}
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
                        let va = Symbol::new(Value::VarArg);
                        results.push(va);
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
                    self.ir.set_stack(StackId::from(i.arga()), global);
                }
                OpCode::Len => {
                    let src = self.ir.get_stack(StackId::from(i.argb()));
                    let val = Symbol::len(src);
                    self.ir.set_stack(StackId::from(i.arga()), val);
                }
                OpCode::Mul => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = Symbol::mul(left, right);
                    self.ir.set_stack(StackId::from(i.arga()), val);
                }
                OpCode::NewTable => {
                    let table = Symbol::new(Value::Table { items: Vec::new() });
                    self.ir.set_stack(StackId::from(i.arga()), table);
                }
                OpCode::TestSet => {
                    let test = self.ir.get_stack(StackId::from(i.argb()));
                    let original = self.ir.get_stack(StackId::from(i.arga()));
                    RefCell::borrow_mut(&original).add_reference(&test);

                    let ijmp = ctx.func.code.get(offset + 1).ok_or(DecompileError::UnexpectedEnd)?;
                    if ijmp.opcode()? != OpCode::Jmp {
                        return Err(DecompileError::ExpectedJmp);
                    }

                    let target = offset as i32 + ijmp.argsbx() + 2;
                    self.tail = Tail::TestSet(ConditionalA {
                        value: test,
                        direction: i.arga() == 0,
                        target_1: offset + 2,
                        target_2: target as usize,
                    }, original);
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
                    self.ir.set_stack(StackId::from(i.arga()), val);
                }
                OpCode::Mod => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = Symbol::modulus(left, right);
                    self.ir.set_stack(StackId::from(i.arga()), val);
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
                    self.ir.set_stack(StackId::from(i.arga()), val);
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

                    let ijmp = ctx.func.code.get(offset + 1).ok_or(DecompileError::UnexpectedEnd)?;
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
                },
                OpCode::Pow => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    self.ir
                        .set_stack(StackId::from(i.arga()), Symbol::pow(left, right));
                }
                OpCode::OpSelf => {
                    let table = self.ir.get_stack(StackId::from(i.argb()));
                    let key = self.stack_or_const(i.argc(), ctx);
                    self.ir.set_stack(StackId::from(i.arga() + 1), table.clone());
                    self.ir.gettable(StackId::from(i.arga()), table, key);
                },
                OpCode::Sub => {
                    let left = self.stack_or_const(i.argb(), ctx);
                    let right = self.stack_or_const(i.argc(), ctx);
                    let val = Symbol::sub(left, right);
                    self.ir.set_stack(StackId::from(i.arga()), val);
                },
                OpCode::Move => {
                    let val = self.ir.get_stack(StackId::from(i.argb()));
                    self.ir.set_stack(StackId::from(i.arga()), val);
                },
                OpCode::Close => {},
                OpCode::LoadNil => {
                    let ra = i.arga();
                    let rb = i.argb();
                    for ri in ra..=rb {
                        self.ir.set_stack(StackId::from(ri), Symbol::nil());
                    }
                },
            }

            if ctx.branches[offset].len() == 1 && ctx.references[offset].len() <= 1 {
                offset = ctx.branches[offset][0];
                assert!(matches!(self.tail, Tail::None));
            } else {
                break;
            }
        }
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

    pub fn make_local(&self) -> Name {
        let idx = self.local_idx.get();
        self.local_idx.set(idx + 1);
        Name::Local(format!("local_{}", idx))
    }
}

#[derive(Debug)]
pub struct FunctionContext {
    func: Rc<Function>,
    nodes: Vec<Rc<Node>>,
    branches: Vec<Vec<usize>>,
    references: Vec<Vec<usize>>,
    root: Rc<RootContext>,
    upvalues: Vec<SymbolRef>,
}

impl FunctionContext {
    fn new(root: Rc<RootContext>, func: Rc<Function>) -> FunctionContext {
        let branches = vec![Vec::new(); func.code.len()];
        let references = vec![Vec::new(); func.code.len()];
        FunctionContext {
            upvalues: (0..func.nups).map(|_| Symbol::upvalue()).collect(),
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

    fn analyze_nodes(&mut self) -> Result<(), DecompileError> {
        let mut heads = HashSet::new();
        heads.insert(0);
        for offset in 1..self.func.code.len() {
            if self.references[offset].len() > 1 {
                heads.insert(offset);
            }
            if self.branches[offset].len() > 1 {
                heads.extend(self.branches[offset].iter());
            }
        }

        for head in heads {
            let mut node = Rc::new(Node::new(head));
            Rc::get_mut(&mut node).unwrap().add_code(self, head)?;
            self.nodes.push(node);
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
    ctx.analyze_nodes()?;

    println!("{:#?}", ctx);

    //let ctx = FunctionContext { func };

    Ok(())
}
