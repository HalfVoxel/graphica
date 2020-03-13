use graphica::main;
#[macro_use]
extern crate log;

fn main() {
    env_logger::init();
    graphica::main::main();
}
