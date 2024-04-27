// inclusion facility

pub use crate::imp::*;

pub use crate::bmor::*;

pub use crate::makeiter::*;

pub use crate::sensitivity::*;

pub use crate::wkmedian::*;

#[derive(Copy, Clone)]
pub enum Algo {
    IMP,
    BMOR,
    CORESET1,
}
