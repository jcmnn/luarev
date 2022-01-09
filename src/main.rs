use std::fs::File;
use std::io::Write;

mod function;
mod reader;
mod decompile;

fn main() {
    let f = function::load_file("/home/jacob/unluapp/luac.out").unwrap();

    {
        let mut file = File::create("/home/jacob/unluapp/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }
}
