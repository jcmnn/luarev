use std::fs::File;
use std::io::Write;

use ir::VariableSolver;
use symeval::NodeFlow;

mod function;
mod reader;
mod ir;
mod symeval;
mod lifter;

fn main() {
    let f = function::load_file("/home/jacob/unluapp/Lotus/Interface/Libs/TimerMgr.lua").unwrap();

    {
        let mut file = File::create("/home/jacob/unluapp/luarev/test.luad").unwrap();
        write!(file, "{}", &f).unwrap();
    }

    let mut solver = VariableSolver::new();

    // decompile::decompile(root, f.clone()).unwrap();
    let tree = lifter::lift(&f, &mut solver).unwrap();
    solver.minimize(&tree);
    let flow = NodeFlow::generate(&tree, &solver);
    println!("{}", flow.source);
    //println!("{:#?}", tree);
    
    //println!("{:#?}", flow);
}
