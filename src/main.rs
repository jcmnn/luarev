use std::fs::File;
use std::io::Write;
use std::rc::Rc;

use decompile::RootContext;

use crate::ir::{NodeFlow, ControlCode};

mod function;
mod reader;
mod ir;
mod decompile;
mod symeval;
mod lifter;

fn main() {
    let f = function::load_file("/home/jacob/luarev/test.luac").unwrap();

    let root = Rc::new(RootContext::new());

    {
        let mut file = File::create("/home/jacob/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }

    // decompile::decompile(root, f.clone()).unwrap();
    let tree = lifter::lift(&f).unwrap();
    //println!("{:#?}", tree);
    println!("Forward: {:?}", tree.next);

    
    //println!("{:#?}", flow);
}
