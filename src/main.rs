use std::fs::File;
use std::io::Write;

use ir::VariableSolver;
use symeval::NodeFlow;

mod function;
mod ir;
mod lifter;
mod reader;
mod symeval;

fn main() {
    let f = function::load_file("/home/jacob/unluapp/Lotus/Scripts/ContainerDropTable.lua").unwrap();
    //let f = function::load_file("/home/jacob/luarev/test.luac").unwrap();

    {
        let mut file = File::create("/home/jacob/unluapp/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }

    let mut solver = VariableSolver::new();

    // decompile::decompile(root, f.clone()).unwrap();
    let tree = lifter::lift(&f, &mut solver).unwrap();
    solver.minimize(&tree);
    let flow = NodeFlow::generate(&tree, &solver);
    {
        let mut file = File::create("/home/jacob/unluapp/luarev/test.dec.lua").unwrap();
        write!(file, "{}", flow.source).unwrap();
    }
    // println!("{}", flow.source);
    //println!("{:#?}", tree);

    //println!("{:#?}", flow);
}
