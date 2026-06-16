//! This module is dedicated to information metrics related to clustering merit.
//! It builds from :
//! - Vinh.N.X Information Theoretic Measures for clustering comparison: [Vinh 2010](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)
//!
//! It defines various quality measures of clustering based on information theory.
//! They are normalized with value in the interval \[0,1\] and some are metrics.  
//! It is also possible to compare different clustering and to quantify how the affectation to items of 2 algorithms are relat
//! ed. It relies on a contingency table.  
//! The various measures are described in the [contingency](./src/contingency.rs) file.
//!
//! We preferentially use *get_nmi_sqrt()*
//! The trait [affectation](./src/affect.rs) is used to make the contingency table aware of of data versus label association.
//!

pub mod affect;
pub mod contingency;

pub use affect::*;
pub use contingency::*;
