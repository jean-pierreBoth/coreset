// inclusion facility

pub use crate::imp::*;

pub use crate::bmor::*;


#[derive(Copy,Clone)]
pub enum Algo {
    IMP,
    BMOR,
}