use lazy_static::lazy_static;

pub mod prelude;

pub mod bmor;
pub mod facility;
pub mod imp;
mod scale;
pub mod sensitivity;

pub mod discrete;
pub mod makeiter;

pub mod clustercore;
pub mod wkmedian;

lazy_static! {
    static ref LOG: u64 = init_log();
}

#[allow(unused)]
// install a logger facility
fn init_log() -> u64 {
    let _res = env_logger::try_init();
    println!("\n ************** initializing logger *****************\n");
    1
}

#[cfg(test)]
mod tests {
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        let _res = env_logger::try_init();
    }
} // end of tests
