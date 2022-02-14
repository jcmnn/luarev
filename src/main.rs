use std::fs::File;
use std::io::Write;
use std::rc::Rc;

mod function;
mod reader;
mod ir;
mod symeval;
mod lifter;

fn main() {
    let f = function::load_file("/home/jacob/unluapp/luarev/test.luac").unwrap();

    {
        let mut file = File::create("/home/jacob/unluapp/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }

    // decompile::decompile(root, f.clone()).unwrap();
    let tree = lifter::lift(&f).unwrap();
    symeval::generate_scope(&tree);
    //println!("{:#?}", tree);
    
    //println!("{:#?}", flow);
}
