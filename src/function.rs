use byteorder::ReadBytesExt;
use int_enum::{IntEnum, IntEnumError};

use crate::reader::LuaReaderExt;
use std::cell::{Cell, RefCell};
use std::fmt::Display;
use std::fmt::{self, Debug};
use std::fs::File;
use std::io::{self, Read, Result as IoResult};
use std::path::Path;
use std::rc::Rc;

/// LVM Opcodes
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, IntEnum)]
pub enum OpCode {
    Test = 0x00,      /*	A C	if not (R(A) <=> C) then pc++			*/
    Close = 0x01,     /*	A 	close all variables in the stack up to (>=) R(A)*/
    SetTable = 0x02,  /*	A B C	R(A)[RK(B)] := RK(C)				*/
    TestSet = 0x03,   /*	A B C	if (R(B) <=> C) then R(A) := R(B) else pc++	*/
    Le = 0x04,        /*	A B C	if ((RK(B) <= RK(C)) ~= A) then pc++  		*/
    SetGlobal = 0x05, /*	A Bx	Gbl[Kst(Bx)] := R(A)				*/
    Sub = 0x06,       /*	A B C	R(A) := RK(B) - RK(C)				*/
    Move = 0x07,      /*	A B	R(A) := R(B)					*/
    Mod = 0x08,       /*	A B C	R(A) := RK(B) % RK(C)				*/
    Jmp = 0x09,       /*	sBx	pc+=sBx					*/
    Eq = 0x0a,        /*	A B C	if ((RK(B) == RK(C)) ~= A) then pc++		*/
    SetCGlobal = 0x0b,
    GetUpval = 0x0c,  /*	A B	R(A) := UpValue[B]				*/
    OpSelf = 0x0d,    /*	A B C	R(A+1) := R(B); R(A) := R(B)[RK(C)]		*/
    Len = 0x0e,       /*	A B	R(A) := length of R(B)				*/
    GetGlobal = 0x0f, /*	A Bx	R(A) := Gbl[Kst(Bx)]				*/
    Closure = 0x10,   /*	A Bx	R(A) := closure(KPROTO[Bx], R(A), ... ,R(A+n))	*/
    Pow = 0x11,       /*	A B C	R(A) := RK(B) ^ RK(C)				*/
    LoadK = 0x12,     /*	A Bx	R(A) := Kst(Bx)					*/
    GetNum = 0x13,
    SetUpval = 0x14, /*	A B	UpValue[B] := R(A)				*/
    ForPrep = 0x15,  /*	A sBx	R(A)-=R(A+2); pc+=sBx				*/
    Div = 0x16,      /*	A B C	R(A) := RK(B) / RK(C)				*/
    Vararg = 0x17,   /*	A B	R(A), R(A+1), ..., R(A+B-1) = vararg		*/
    TForLoop = 0x18, /*	A C	R(A+3), ... ,R(A+2+C) := R(A)(R(A+1), R(A+2));
                     if R(A+3) ~= nil then R(A+2)=R(A+3) else pc++	*/
    TailCall = 0x19, /*	A B C	return R(A)(R(A+1), ... ,R(A+B-1))		*/
    Mul = 0x1a,      /*	A B C	R(A) := RK(B) * RK(C)				*/
    LoadNil = 0x1b,  /*	A B	R(A) := ... := R(B) := nil			*/
    Lt = 0x1c,       /*	A B C	if ((RK(B) <  RK(C)) ~= A) then pc++  		*/
    GetTable = 0x1d, /*	A B C	R(A) := R(B)[RK(C)]				*/
    Return = 0x1e,   /*	A B	return R(A), ... ,R(A+B-2)	(see note)	*/
    Not = 0x1f,      /*	A B	R(A) := not R(B)				*/
    NewTable = 0x20, /*	A B C	R(A) := {} (size = B,C)				*/
    LoadBool = 0x21, /*	A B C	R(A) := (Bool)B; if (C) pc++			*/
    Call = 0x22,     /*	A B C	R(A), ... ,R(A+C-2) := R(A)(R(A+1), ... ,R(A+B-1)) */
    // 0x23 seems like xor?
    Unm = 0x23,     /*	A B	R(A) := -R(B)					*/
    Concat = 0x24,  /*	A B C	R(A) := R(B).. ... ..R(C)			*/
    SetList = 0x25, /*	A B C	R(A)[(C-1)*FPF+i] := R(A+i), 1 <= i <= B	*/
    ForLoop = 0x26, /*	A sBx	R(A)+=R(A+2);
                    if R(A) <?= R(A+1) then { pc+=sBx; R(A+3)=R(A) }*/
    Add = 0x27, /*	A B C	R(A) := RK(B) + RK(C)				*/
}

/// Enum representing a constant ID or Register (stack) ID
#[derive(Debug)]
pub enum ConstReg {
    Const(u32),
    Reg(u32),
}

impl ConstReg {
    pub fn new(n: u32) -> ConstReg {
        if n & (1 << 8) != 0 {
            ConstReg::Const(n & 0xFF)
        } else {
            ConstReg::Reg(n)
        }
    }
}

impl Display for ConstReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstReg::Const(k) => write!(f, "K{}", k),
            ConstReg::Reg(r) => write!(f, "R{}", r),
        }
    }
}

#[derive(Clone, Copy)]
pub struct LvmInstruction(u32);

impl LvmInstruction {
    pub fn opcode(&self) -> Result<OpCode, IntEnumError<OpCode>> {
        OpCode::from_int(self.0 & 0x3F)
    }

    pub fn raw(&self) -> u32 {
        self.0
    }

    pub fn arga(&self) -> u32 {
        (self.0 >> 6) & 0xFF
    }

    pub fn argb(&self) -> u32 {
        self.0 >> 23
    }

    pub fn argc(&self) -> u32 {
        (self.0 >> 14) & 0x1FF
    }

    pub fn argb_rk(&self) -> ConstReg {
        ConstReg::new(self.argb())
    }

    pub fn argc_rk(&self) -> ConstReg {
        ConstReg::new(self.argc())
    }

    pub fn argbx(&self) -> u32 {
        self.0 >> 14
    }

    pub fn argsbx(&self) -> i32 {
        (self.argbx() as i32) - (((1 << 18) - 1) >> 1)
    }

    pub fn number(&self) -> f32 {
        let b = self.argbx() << 14;
        let bytes: [u8; 4] = b.to_be_bytes();
        f32::from_be_bytes(bytes)
    }
}

impl Debug for LvmInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for LvmInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.opcode().map_err(|_| fmt::Error)? {
            OpCode::TailCall => write!(
                f,
                "TAILCALL R{} {} {}",
                self.arga(),
                self.argb(),
                self.argc()
            ),
            OpCode::Closure => write!(f, "CLOSURE R{} K{}", self.arga(), self.argbx()),
            OpCode::Eq => write!(
                f,
                "EQ {} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::Div => write!(
                f,
                "DIV R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::GetNum => write!(
                f,
                "GETNUM {}",
                f32::from_ne_bytes((self.argbx() << 14).to_ne_bytes())
            ),
            OpCode::Concat => write!(
                f,
                "CONCAT R{} R{} R{}",
                self.arga(),
                self.argb(),
                self.argc()
            ),
            OpCode::GetTable => write!(
                f,
                "GETTABLE R{} R{} {}",
                self.arga(),
                self.argb(),
                self.argc_rk()
            ),
            OpCode::SetList => write!(
                f,
                "SETLIST R{} {} {}",
                self.arga(),
                self.argb(),
                self.argc()
            ),
            OpCode::LoadK => write!(f, "LOADK R{} K{}", self.arga(), self.argbx()),
            OpCode::SetGlobal => write!(f, "SETGLOBAL R{} K{}", self.arga(), self.argbx()),
            OpCode::Jmp => write!(f, "JMP {}", self.argsbx()),
            OpCode::TForLoop => write!(f, "TFORLOOP R{} {}", self.arga(), self.argc()),
            OpCode::SetUpval => write!(f, "SETUPVAL R{} {}", self.arga(), self.argb()),
            OpCode::Not => write!(f, "NOT R{} R{}", self.arga(), self.argb()),
            OpCode::Vararg => write!(f, "VARARG R{} {}", self.arga(), self.argb() as i32 - 1),
            OpCode::GetUpval => write!(f, "GETUPVAL R{} {}", self.arga(), self.argb()),
            OpCode::Add => write!(
                f,
                "ADD R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::Return => write!(f, "RETURN R{} {}", self.arga(), self.argb() as i32 - 1),
            OpCode::GetGlobal => write!(f, "GETGLOBAL R{} K{}", self.arga(), self.argbx()),
            OpCode::Len => write!(f, "LEN R{} R{}", self.arga(), self.argb()),
            OpCode::Mul => write!(
                f,
                "MUL R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::NewTable => write!(
                f,
                "NEWTABLE R{} {} {}",
                self.arga(),
                self.argb(),
                self.argc()
            ),
            OpCode::TestSet => write!(
                f,
                "TESTSET R{} R{} {}",
                self.arga(),
                self.argb(),
                self.argc()
            ),
            OpCode::SetTable => write!(
                f,
                "SETTABLE R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::Unm => write!(f, "UNM R{} R{}", self.arga(), self.argb()),
            OpCode::Mod => write!(
                f,
                "MOD R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::Lt => write!(
                f,
                "LT {} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::ForLoop => write!(f, "FORLOOP R{} {}", self.arga(), self.argsbx()),
            OpCode::Call => write!(
                f,
                "CALL R{} {} {}",
                self.arga(),
                self.argb() as i32 - 1,
                self.argc() as i32 - 1
            ),
            OpCode::Le => write!(
                f,
                "LE {} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::LoadBool => write!(
                f,
                "LOADBOOL R{} {} {}",
                self.arga(),
                self.argb(),
                self.argc()
            ),
            OpCode::ForPrep => write!(f, "FORPREP R{} {}", self.arga(), self.argsbx()),
            OpCode::SetCGlobal => write!(f, "SETCGLOBAL R{} K{}", self.arga(), self.argbx()),
            OpCode::Test => write!(f, "TEST R{} {}", self.arga(), self.argc()),
            OpCode::Pow => write!(
                f,
                "POW R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::OpSelf => write!(
                f,
                "SELF R{} R{} {}",
                self.arga(),
                self.argb(),
                self.argc_rk()
            ),
            OpCode::Sub => write!(
                f,
                "SUB R{} {} {}",
                self.arga(),
                self.argb_rk(),
                self.argc_rk()
            ),
            OpCode::Move => write!(f, "MOVE R{} R{}", self.arga(), self.argb()),
            OpCode::Close => write!(f, "CLOSE R{}", self.arga()),
            OpCode::LoadNil => write!(f, "LOADNIL R{} R{}", self.arga(), self.argb()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Constant {
    Nil,
    Boolean(u32),
    Number(f32),
    String(String),
}

impl Constant {
    pub fn write_global(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Nil => write!(f, "nil"),
            Constant::Boolean(b) => write!(f, "coded_global({:#X})", b),
            Constant::Number(n) => write!(f, "{}", n),
            Constant::String(s) => write!(f, "{}", s),
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Nil => write!(f, "nil"),
            Constant::Boolean(b) => write!(f, "{:#X}", b),
            Constant::Number(n) => write!(f, "{}", n),
            Constant::String(s) => write!(f, "{:?}", s),
        }
    }
}

#[derive(Debug)]
pub struct Function {
    pub source: String,
    pub line_defined: u32,
    pub last_line_defined: u32,
    pub nups: u8,
    pub is_vararg: u8,
    pub max_stack_size: u8,
    pub code: Vec<LvmInstruction>,
    pub constants: Vec<Constant>,
    pub num_params: usize,
    pub closures: Vec<Rc<Function>>,
}

fn load_code<R: Read>(mut rdr: R) -> IoResult<Vec<LvmInstruction>> {
    let code_size = rdr.read_varint()?;
    let mut code = Vec::with_capacity(code_size as usize);
    for _ in 0..code_size {
        code.push(LvmInstruction(rdr.read_uinteger()?));
    }
    Ok(code)
}

fn load_constants<R: Read>(mut rdr: R) -> IoResult<Vec<Constant>> {
    let size = rdr.read_varint()?;
    let mut constants = Vec::with_capacity(size as usize);
    for _ in 0..size {
        match rdr.read_byte()? {
            0 => constants.push(Constant::Nil),
            1 => constants.push(Constant::Boolean(rdr.read_uinteger()?)),
            3 => constants.push(Constant::Number(rdr.read_number()?)),
            4 => constants.push(Constant::String(rdr.read_string()?)),
            _ => panic!("Invalid constant"),
        };
    }
    Ok(constants)
}

fn load_closures<R: Read>(rdr: &mut R) -> IoResult<Vec<Rc<Function>>> {
    let size = rdr.read_varint()?;
    let mut closures = Vec::with_capacity(size as usize);
    for _ in 0..size {
        closures.push(Function::load(&mut *rdr)?);
    }
    Ok(closures)
}

fn load_debug<R: Read>(mut rdr: R) -> IoResult<()> {
    let lineinfo_size = rdr.read_varint()? as usize;
    let _lineinfo = (0..lineinfo_size)
        .map(|_| rdr.read_uinteger())
        .collect::<IoResult<Vec<u32>>>()?;
    // Read local variables
    for _ in 0..rdr.read_varint()? {
        rdr.read_string()?;
        rdr.read_varint()?;
        rdr.read_varint()?;
    }
    // Read upvalues
    for _ in 0..rdr.read_varint()? {
        rdr.read_string()?;
    }

    Ok(())
}

impl Function {
    pub fn load<R: Read>(rdr: &mut R) -> IoResult<Rc<Function>> {
        let source = rdr.read_string()?;
        let line_defined = rdr.read_varint()?;
        let last_line_defined = rdr.read_varint()?;
        let nups = rdr.read_byte()?;
        let num_params = rdr.read_byte()?;
        let is_vararg = rdr.read_byte()?;
        let max_stack_size = rdr.read_byte()?;

        // Load code
        let code = load_code(&mut *rdr)?;
        let constants = load_constants(&mut *rdr)?;
        let closures = load_closures(&mut *rdr)?;
        load_debug(&mut *rdr)?;

        Ok(Rc::new(Function {
            source,
            line_defined,
            last_line_defined,
            nups,
            is_vararg,
            max_stack_size,
            code,
            constants,
            num_params: num_params as usize,
            closures,
        }))
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "-- Source: {}", self.source)?;
        writeln!(f, "-- Line defined: {}", self.line_defined)?;
        writeln!(f, "-- Last line defined: {}", self.last_line_defined)?;
        writeln!(f, "-- Number upvalues: {}", self.nups)?;
        writeln!(f, "-- Is vararg: {}", self.is_vararg)?;
        writeln!(f, "-- Max stack size: {}", self.max_stack_size)?;
        writeln!(f, "-- Params: {}", self.num_params)?;
        writeln!(f, "\n-- Constants:")?;
        for (i, c) in self.constants.iter().enumerate() {
            writeln!(f, "{}: {}", i, c)?;
        }

        writeln!(f, "\n.start")?;
        for (index, i) in self.code.iter().enumerate() {
            writeln!(f, "{:>3} {}", index, i)?;
        }

        writeln!(f, "\n-- Closures:")?;
        for (i, c) in self.closures.iter().enumerate() {
            writeln!(f, "-- Closure {}\n{}\n", i, c)?;
        }

        Ok(())
    }
}

pub fn load_file<P: AsRef<Path>>(path: P) -> IoResult<Rc<Function>> {
    let mut file = File::open(path)?;
    if file.read_byte()? != 0x7F {
        return Err(io::Error::new(io::ErrorKind::Other, "invalid header"));
    }
    Function::load(&mut file)
}
