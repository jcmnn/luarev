use std::fs::File;
use std::io::Write;
use std::rc::Rc;

use decompile::RootContext;

mod function;
mod reader;
mod ir;
mod decompile;
mod symeval;

fn main() {
    let f = function::load_file("/home/jacob/unluapp/luarev/test.luac").unwrap();

    let root = Rc::new(RootContext::new());

    {
        let mut file = File::create("/home/jacob/unluapp/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }

    decompile::decompile(root, f.clone()).unwrap();
}
