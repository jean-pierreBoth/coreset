//! Implementation of the Mettu-Plaxton algorithm as analyzed in 
//!  Facility Location in sublinear time 
//!       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
//!       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)
//! 
//!  The data we will run on are essentially Vec<T> where T can be anything as long as we have a distance on Vec<T> provided by the hnsw crate
//!  see [hnsw-dist](https://docs.rs/hnsw_rs/0.1.19/hnsw_rs/dist/index.html)
//! 
//! Data or Distance must be scaled so that nearest neighbour of a point are at a distance really less than 1. as uniform cost is an explicit hypothesis 
//! of the paper.

#![allow(unused)]
use anyhow::{anyhow, Result};

use rayon::prelude::*;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use rand::distributions::{Distribution,Uniform};

use hnsw_rs::dist::*;

/// a facility is a center in coreset approximation
pub struct Facility<T: Send+Sync+Clone> {
    /// rank 
    dataid : usize,

    position : Vec<T>,
}

impl <T:Send+Sync+Clone> Facility<T> {
    ///
    pub fn new(dataid : usize, position : &Vec<T>) -> Self {
        Facility{dataid, position : (*position).clone()}
    }

    pub fn get_position(&self) -> &Vec<T> {
        &self.position
    }

    pub fn get_id(&self) -> usize {
        self.dataid
    }
}  // end of impl block Facility

//==================================================================================


/// describes the list of facility (or centers created)
pub struct Facilities<T : Send+Sync+Clone> {
    centers : Vec<Facility<T>>
}

impl <T:Send+Sync+Clone> Facilities<T> {

    /// to be allocated , size should be log(nb_data)
    pub fn new(size : usize) -> Self {
        let centers = Vec::<Facility<T>>::with_capacity(size);
        Facilities{centers}
    }

    // return true if there is a facility around point at distance less than dmax
    fn match_point<Dist : Distance<T>>(&self, point : &Vec<T>, dmax : f32, distance : &Dist) -> bool {
        //
        for f in &self.centers {
            if distance.eval(f.get_position(), point) <= dmax {
                return true;
            }
        }
        return false;
    } // end of match_facility

    fn insert(&mut self, facility : Facility<T>) {
        self.centers.push(facility);
    }
} // end of impl block Facilities

//==================================================================================

pub struct MettuPlaxton <'b, T: Send+Sync> {
    //
    nb_data : usize,
    //
    data : &'b Vec<Vec<T>>,
    // j is integer value of log data.len()
    j       : u32,
    //
    rng : Xoshiro256PlusPlus,
} // end of struct MettuPlaxton



impl <'b, T:Send+Sync+Clone> MettuPlaxton<'b,T> {

    pub fn new(data : &'b Vec<Vec<T>>) -> Self {
        let nb_data = data.len();
        let j = data.len().ilog2() as u32;
        MettuPlaxton{data, nb_data, j, rng : Xoshiro256PlusPlus::seed_from_u64(123)}
    }

    pub fn get_nb_data(&self) -> usize {
        self.nb_data
    }

    // estimate cardinal around point in a radius of 2^-j. 
    // sample K = c * 2^{-j} * n log n points to estimate N = |B(point , 2−j )|
    // return n * N /K
    // running time O(r * n * log(n)).
    // to be called in //
    fn estimate_ball_cardinal<Dist : Distance<T>>(&self, (ip, point) : (usize,  &Vec<T>), distance : &Dist) -> (usize, f32) 
        where Dist : Sync {
        //
        let c = 2.;
        let mut j_tmp = self.j;
        // at beginning nb_sample = c * self.j i.e c * log(nb_data) and double at each iteration
        let mut rng = self.rng.clone();
        let mut iter_num = 0;
        let mut nb_point_done = 0;
        rng.jump();
        let unif = Uniform::<usize>::new(0, self.nb_data);
        let r : f32 = loop {
            let mut r_test = 1.0f32/ 2_u32.pow(j_tmp) as f32;
            let nb_sample_f : f64 = c * (self.nb_data as f64 * r_test as f64) * self.j as f64;
            let nb_sample : u32 = nb_sample_f.trunc() as u32;
            log::trace!("estimate_ball_cardinal nb_sample : {:?}", nb_sample);
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
                log::debug!("estimate_ball_cardinal for point {:?} ; nb_iter = {:?}, cardinal : {:?}", ip, iter_num, nb_in);
                // an estimator of radius is 1/2^j
                break (self.nb_data * nb_in) as f32 /  nb_sample as f32;
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



    /// construct centers (facilities) for distance 
    pub fn construct_centers<Dist : Distance<T>>(&self, distance : &Dist) -> Facilities<T>
         where Dist : Send + Sync {
        //
        let mut facilities = Facilities::<T>::new(self.j as usize);
        // estimate radii in //
        let cardinals : Vec<(usize,f32)> = (0..self.nb_data).into_par_iter().map(|i| self.estimate_ball_cardinal((i, &self.data[i]), distance)).collect();
        log::debug!("estimate_ball_cardinal done");
        // sort radii
        let mut radii : Vec<(usize,f32)> = cardinals.iter().map(|(i,c)|  (*i, 1./c)).collect();
        radii.sort_unstable_by(|it1, it2| it1.1.partial_cmp(&it2.1).unwrap());
        assert!(radii.first().unwrap().1 <= radii.last().unwrap().1);
        // facility allocation, loop on data, check for existing facility around each point
        for p in radii.iter() {
            let matched = facilities.match_point::<Dist>(&self.data[p.0],  2. * p.1, distance);
            if !matched {
                // we inset a facility
                let facility = Facility::new(p.0, &self.data[p.0]);
                facilities.insert(facility);
                log::debug!("inserted facility at {:?}, radius : {:.3e}", p.0, p.1);
            }
        }
        //
        return facilities;
    } // end of construct_centers


} // end of impl block