use graphica::something;
#[macro_use]
extern crate log;

fn main() {
    env_logger::init();
    graphica::something::main();
}
