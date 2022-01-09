use std::io;
use std::io::Result as IoResult;

use byteorder::{LittleEndian, ReadBytesExt};

pub trait LuaReaderExt: io::Read {
    fn read_number(&mut self) -> IoResult<f32> {
        self.read_f32::<LittleEndian>()
    }

    fn read_varint(&mut self) -> IoResult<u32> {
        let mut x = 0_u32;
        let mut shift = 0;
        while shift <= 7 * 4 {
            let b = self.read_u8()? as u32;
            x = ((b & 0x7F) << shift) | x;
            shift += 7;
            if b < 0x80 {
                break;
            }
        }
        Ok(x)
    }

    fn read_uinteger(&mut self) -> IoResult<u32> {
        self.read_u32::<LittleEndian>()
    }

    fn read_byte(&mut self) -> IoResult<u8> {
        self.read_u8()
    }

    fn read_string(&mut self) -> IoResult<String> {
        let len = self.read_varint()? as usize;
        if len == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0_u8; len];
        self.read_exact(&mut buf)?;
        buf.pop();
        Ok(String::from_utf8(buf).unwrap())
    }
}

impl<T> LuaReaderExt for T where T: io::Read {}