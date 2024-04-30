//! This binary is dedicated to coreset computations on data stored in Hnsw created by crate [hnsw_rs](https://crates.io/crates/hnsw_rs)
//!
//!

#![allow(unused)]

use std::fs::OpenOptions;
use std::path::PathBuf;

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use coreset::prelude::*;
use std::iter::Iterator;

use std::default::Default;

//========================================

/// Coreset parameters
struct CoresetParams {
    beta: f32,
    gamma: f32,
}

impl CoresetParams {
    fn new(beta: f32, gamma: f32) -> CoresetParams {
        CoresetParams { beta, gamma }
    }
    //
    fn get_beta(&self) -> f32 {
        self.beta
    }

    //
    fn get_gamma(&self) -> f32 {
        self.gamma
    }
}

impl Default for CoresetParams {
    fn default() -> Self {
        CoresetParams {
            beta: 2.,
            gamma: 2.,
        }
    }
}

//==========================================

fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    log::info!("running hnswcore");
}
