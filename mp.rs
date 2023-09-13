//! Implementation of the Mettu-Plaxton algorithm as analyzed in 
//!  Facility Location in sublinear time 
//!       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
//!       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)
//! 
//!  The data we will run on are essentially Vec<T> where T can be anything as long as we have a distance on Vec<T> provided by the hnsw crate
//!  see [hnsw-dist](https://docs.rs/hnsw_rs/0.1.19/hnsw_rs/dist/index.html)

use hnsw_rs::dist;

