use std::fs::File;
use std::io::Write;
use std::rc::Rc;

use decompile::RootContext;

mod function;
mod reader;
mod ir;
mod decompile;

fn main() {
    let f = function::load_file("/home/jacob/luarev/test.luac").unwrap();

    let root = Rc::new(RootContext::new());
    decompile::decompile(root, f.clone()).unwrap();

    {
        let mut file = File::create("/home/jacob/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }
}
