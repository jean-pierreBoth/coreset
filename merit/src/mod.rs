//! This module is dedicated to information metrics related to clustering merit
//! It builds from :
//! - Vinh.N.X Information Theoretic Measures for clustering comparison: [Vinh 2010](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)
//!

pub mod affect;
pub mod contingency;

pub use affect::*;
pub use contingency::*;
