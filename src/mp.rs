//! Implementation of the Mettu-Plaxton algorithm as analyzed in 
//!  Facility Location in sublinear time 
//!       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
//!       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)
//! 
//!  The data we will run on are essentially Vec<T> where T can be anything as long as we have a distance on Vec<T> provided by the hnsw crate
//!  see [hnsw-dist](https://docs.rs/hnsw_rs/0.1.19/hnsw_rs/dist/index.html)

#![allow(unused)]
use anyhow::{anyhow, Result};

use parking_lot::{RwLock, Mutex, RwLockReadGuard};
use std::sync::Arc;
use rayon::prelude::*;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use rand::distributions::{Distribution,Uniform};

use hnsw_rs::dist::*;

struct MettuPlaxton <'b, T: Send+Sync> {
    //
    nb_data : usize,
    //
    data : &'b Vec<Vec<T>>,
    // j is integer value of log data.len()
    j       : u32,
    //
    rng : Xoshiro256PlusPlus,
} // end of struct MettuPlaxton



impl <'b, T:Send+Sync> MettuPlaxton<'b,T> {

    pub fn new(data : &'b Vec<Vec<T>>) -> Self {
        let nb_data = data.len();
        let j = data.len().ilog2() as u32;
        MettuPlaxton{data, nb_data, j, rng : Xoshiro256PlusPlus::seed_from_u64(123)}
    }


    // estimate cardinal around point in a radius of 2^-j. 
    // sample K = c * 2^{-j} * n log n points to estimate N = |B(point , 2−j )|
    // return n * N /K
    // to be called in //
    fn estimate_ball_cardinal<Dist : Distance<T>>(&self, (ip, point) : (usize,  &Vec<T>), distance : Dist) -> (usize, f64) {
        //
        let c = 100.;
        let mut j_tmp = self.j;
        // at beginning nb_sample = c * self.j i.e c * log(nb_data) and double at each iteration
        let mut rng = self.rng.clone();
        let mut iter_num = 0;
        rng.jump();
        let unif = Uniform::<usize>::new(0, self.nb_data);
        let r : f64 = loop {
            let mut r_test = 1.0f32/ 2_u32.pow(j_tmp) as f32;
            let nb_sample_f : f64 = c * (self.nb_data as f64 * r_test as f64) * self.j as f64;
            let nb_sample : u32 = nb_sample_f.trunc() as u32;
            let mut nb_in = 0;
            for i in 0..nb_sample {
                // sample and compute distance to point ip
                let k = unif.sample(&mut rng);
                let dist = distance.eval(point, &self.data[k]);
                if dist  <= r_test {
                    nb_in += 1;
                }
            }
            // 
            if nb_in >= 2_usize.pow(j_tmp) {
                log::debug!("estimate_ball_cardinal for point {:?} ; nb_iter = {:?}", ip, iter_num);
                break (self.nb_data * nb_in) as f64 /  nb_sample as f64;
            }
            else {
                if j_tmp < 1 {
                    log::error!("error in estimate_ball_cardinal, j_tmp becomes negative");
                    std::process::exit(1);
                }
                j_tmp = j_tmp - 1;
                iter_num += 1;
            }
        };
        return (ip,r);
    }


    fn estimate_radii<Dist : Distance<T>>(&self) -> Result<Vec<Vec<(usize, f32)>>, anyhow::Error >{

        return Err(anyhow!("not yet implemented"));
    }
} // end of impl block