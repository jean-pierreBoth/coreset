//! Implementation of the Mettu-Plaxton algorithm as analyzed in 
//!  Facility Location in sublinear time 
//!       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
//!       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)
//! 
//! 
//!  Optimal Time bounds for approximate Clustering 
//!     Mettu-Plaxton 2004.  [mettu-plaxton-2](https://link.springer.com/article/10.1023/b:mach.0000033114.18632.e0)
//! 
//! 
//!  The algorithm computes an $\alpha$, $\beta$ k-median approximation that can be used as input
//!  to coreset computations. The  Badoiu paper and Mettu-Plaxtion 2004 paper builds upon the Mettu-Plaxton paper : 
//!  The online median problem. SIAM J. Computing 2003.
//! 
//!  The data we will run on are essentially Vec<T>where T can be anything as long as we have a distance on Vec<T> provided by the hnsw crate
//!  see [hnsw-dist](https://docs.rs/hnsw_rs/0.1.19/hnsw_rs/dist/index.html)
//! 
//! Data or Distance must be scaled so that nearest neighbour of a point are at a distance really less than 1. as uniform cost is an explicit hypothesis 
//! of the paper.

#![allow(unused)]
use anyhow::{anyhow, Result};

use rayon::prelude::*;
use parking_lot::RwLock;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use rand::distributions::{Distribution,Uniform};
use quantiles::ckms::CKMS;     // we could use also greenwald_khanna

use hnsw_rs::dist::*;

use crate::scale::*;
use crate::facility::*;


//==================================================================================


/// Mettu-Plaxton algorithm with Indyk simplification as described in  [indyk](https://people.csail.mit.edu/indyk/fl.pdf)
/// If data have weights attached, we split the data in subsets of approximately equal weights as suggested in :  
/// Chen K., On Coresets Kmedian Clustering MetricSpaces And Applications 2009 Siam J. Computing
pub struct MettuPlaxton <'b, T: Send+Sync, Dist : Distance<T>> {
    //
    nb_data : usize,
    //
    data : &'b Vec<Vec<T>>,
    // j is integer value of log data.len()
    j       : u32,
    //
    distance : Dist,
    //
    rng : Xoshiro256PlusPlus,
} // end of struct MettuPlaxton



impl <'b, T:Send+Sync+Clone, Dist : Distance<T>> MettuPlaxton<'b,T, Dist> {

    /// initialization for unweighted data
    pub fn new(data : &'b Vec<Vec<T>>, distance : Dist) -> Self {
        let nb_data: usize = data.len();
        let j: u32 = data.len().ilog2() as u32;
        //
        MettuPlaxton{nb_data, data, j, distance, rng : Xoshiro256PlusPlus::seed_from_u64(123)}
    }


    pub fn get_nb_data(&self) -> usize {
        self.nb_data
    }

    // estimate cardinal around point in a radius of 2^-j. 
    // sample K = c * 2^{-j} * n log n points to estimate N = |B(point , 2−j )|
    // return n * N /K
    // running time O(r * n * log(n)).
    // to be called in //
    fn estimate_ball_cardinal(&self, (ip, point) : (usize,  &Vec<T>), scale : f32) -> (usize, f32) 
        where Dist : Sync {
        //
        let c = 2.;
        let j_shift = scale.log2() as usize;
        let mut j_tmp = self.j ;
        // at beginning nb_sample = c * self.j i.e c * log(nb_data) and double at each iteration
        let mut rng = self.rng.clone();
        let mut iter_num = 0;
        let mut nb_point_done = 0;
        rng.jump();
        let unif = Uniform::<usize>::new(0, self.nb_data);
        let r : f32 = loop {
            let mut r_test = scale / 2_u32.pow(j_tmp) as f32;
            let nb_sample_f : f32 = c * (self.nb_data as f32 * r_test / scale) / self.j as f32;
            let nb_sample : u32 = nb_sample_f.trunc() as u32;
            log::trace!("estimate_ball_cardinal nb_sample : {:?}", nb_sample);
            let mut nb_in = 1;  // count center in!
            for i in 0..nb_sample {
                // sample and compute distance to point ip
                let k = unif.sample(&mut rng);
                let dist = self.distance.eval(point, &self.data[k]);
                if dist  <= r_test {
                    nb_in += 1;
                }
            }
            // 
            if nb_in >= 2_usize.pow(j_tmp) {
                log::debug!("estimate_ball_cardinal for point {:?} ; nb_iter = {:?}, cardinal : {:?}, radius : {:.3e}", ip, iter_num, nb_in, r_test);
                // an estimator of radius is 1/2^j
                break r_test;
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



    /// construct centers (facilities) for a given distance and returns allocated facilities (or centers)
    pub fn construct_centers(&self) -> Facilities<T, Dist>
         where Dist : Send + Sync + Clone {
        // get scales
        let q_dist = scale_estimation(1_000_000, self.data, &self.distance);
        let d_range = (q_dist.query(0.0001).unwrap().1, q_dist.query(0.95).unwrap().1);
        let d_median = q_dist.query(0.5).unwrap().1;
        log::info!("dist median : {:.3e}",d_median);
        //
        let mut facilities = Facilities::<T, Dist>::new(self.j as usize, self.distance.clone());
        // estimate radii in //
        let mut radii : Vec<(usize,f32)> = (0..self.nb_data).into_par_iter().map(|i| self.estimate_ball_cardinal((i, &self.data[i]), d_median)).collect();
        log::debug!("estimate_ball_cardinal done");
        // sort radii
        radii.sort_unstable_by(|it1, it2| it1.1.partial_cmp(&it2.1).unwrap());
        assert!(radii.first().unwrap().1 <= radii.last().unwrap().1);
        // facility allocation, loop on data, check for existing facility around each point
        for p in radii.iter() {
            let matched = facilities.match_point(&self.data[p.0],  2. * p.1, &self.distance);
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




    // affect each point to a facility.
    pub fn compute_cost_serial(&self, facilities : &Facilities<T, Dist>, data : &Vec<Vec<T>>, distance : &Dist)
        where Dist : Send + Sync {
            //
        log::info!("MettuPlaxton computing costs ...");
        let nb_facility = facilities.len();
        for i in 0..data.len() {
            let mut affectation = Vec::<(usize, f32)>::with_capacity(nb_facility);
            for j in 0..nb_facility {
                let facility = facilities.get_facility(j).unwrap().read();
                let (jf,dist) = (j, distance.eval(&data[i], facility.get_position()));
                affectation.push((jf,dist));
            }
            // sort
            affectation.sort_unstable_by(|it1, it2| it1.1.partial_cmp(&it2.1).unwrap());
            // point i is affected to affectation[0]
            let f_rank = affectation[0].0;
            let dist = affectation[0].1;
            let mut facility = facilities.get_facility(f_rank).unwrap();
            facility.write().insert(1., dist);

        }
        //
        let mut total_weight = 0.;
        log::info!("nb facilities  : {:?}", nb_facility);
        for i in 0..nb_facility {
            let facility = facilities.get_facility(i).unwrap().read();
            total_weight += facility.get_weight();
            facilities.get_facility(i).unwrap().read().log();
        }
        log::info!("weight dispatched into facilities : {:.5e}", total_weight);
    } // end of compute_cost_serial



    // affect each point to a facility.
    pub fn compute_cost_parallel(&self, facilities : &Facilities<T, Dist>, data : &Vec<Vec<T>>, distance : &Dist)
        where Dist : Send + Sync {
            //
        log::info!("MettuPlaxton computing costs ...");
        //
        let nb_facility = facilities.len();
        let affect = | i : usize| -> u8 {
            let mut affectation = Vec::<(usize, f32)>::with_capacity(nb_facility);
            for j in 0..nb_facility {
                let facility = facilities.get_facility(j).unwrap().read();
                let (jf,dist) = (j, distance.eval(&data[i], facility.get_position()));
                affectation.push((jf,dist));
            }
            // sort
            affectation.sort_unstable_by(|it1, it2| it1.1.partial_cmp(&it2.1).unwrap());
            // point i is affected to affectation[0]
            let f_rank = affectation[0].0;
            let dist = affectation[0].1;
            let mut facility = facilities.get_facility(f_rank).unwrap();
            facility.write().insert(1., dist);
            //
            return 1;           
        };
        //
        let res : Vec<u8>= (0..data.len()).into_par_iter().map(|i| affect(i)).collect();
        //
        //
        for i in 0..nb_facility {
            facilities.get_facility(i).unwrap().read().log();
        }
    } // end of  compute_cost  



    pub fn compute_cost(&self, facilities : &Facilities<T,Dist>, data : &Vec<Vec<T>>, proba : f64)
    where Dist : Send + Sync {
        //
        if data.len() > 1_000_000 {
            self.compute_cost_parallel(facilities, data, &self.distance);
        }
        else {
            self.compute_cost_serial(facilities, data, &self.distance);
        }
        //
        facilities.cross_distances(proba);
    } // end of compute_cost


} // end of impl block


//======================================================================================


/// Mettu-Plaxton online median algorithm Siam 2003 [online-median](https://epubs.siam.org/doi/10.1137/S0097539701383443)
/// 
/// This algorithm can handle weighted data but its complexity is O(n²) so it 
/// is more adapted to final processing of Coreset algorithms (bmor algos   
/// or blackbox in Chen K., On Coresets Kmedian ClusteringM etricSpaces And Applications 2009 Siam J. Computing)
pub struct WeightedMettuPlaxton <'b, T: Send+Sync, Dist : Distance<T>> {
    //
    nb_data : usize,
    //
    data : &'b Vec<Vec<T>>,
    //
    weights : &'b Vec<f32>,
    // j is integer value of log data.len()
    j       : u32,
    //
    distance : Dist,
    //
    rng : Xoshiro256PlusPlus,
} // end of struct WeightedMettuPlaxton


impl <'b, T:Send+Sync+Clone, Dist : Distance<T> + Send + Sync> WeightedMettuPlaxton<'b,T, Dist> {

    /// initialization for unweighted data
    pub fn new(data : &'b Vec<Vec<T>>, weights : &'b Vec<f32>, distance : Dist) -> Self {
        let nb_data: usize = data.len();
        let j: u32 = data.len().ilog2() as u32;
        //
        WeightedMettuPlaxton{nb_data, data, weights, j, distance, rng : Xoshiro256PlusPlus::seed_from_u64(123)}
    }


    pub fn get_nb_data(&self) -> usize {
        self.nb_data
    }

    // compute all cross distances
    fn compute_all_dists(&self) -> Vec<RwLock<Vec<f32>>> {
        let mut dists : Vec<RwLock<Vec<f32>>> = Vec::<RwLock<Vec<f32>>>::with_capacity(self.nb_data);
        //
        for i in 0..self.nb_data {
            let d :Vec<f32> = (0..self.nb_data).into_iter().map(|_| -1.).collect();
            dists.push(RwLock::new(d));
        }

        let threshold = 1000;
        let compute_for_i = | i : usize |  {
            let mut dist_i : Vec<f32> = (0..self.nb_data).into_iter().map(|_| -1.).collect();
            for j in 0..i {
                let dist = dists[j].read()[i];
                let dist_i_j = if dist < 0. {
                    self.distance.eval(&self.data[i], &self.data[j])
                }
                else {
                    dist
                };
                dist_i[j] = dist_i_j;
            }; // end of for j
            *dists[i].write() = dist_i;
            return 1;
        };
        //
        let _res : Vec<i32> = (0..self.nb_data).into_par_iter().map(|i| compute_for_i(i)).collect();
        // now we have all dists
        return dists
    } // end of compute_all_dists


    fn compute_ball_radius(&self, ball : usize, value : f32, dists  : &RwLock<Vec<f32>>) -> f32 {
        //
        log::debug!("WeightedMettuPlaxton compute_ball_radius , value to match {:.3e}", value);
        //
        // we sort distances
        let mut indexed_dist : Vec<(usize, f32)> = (0..self.nb_data).into_iter().zip(dists.read().iter()).map(|(i, f)| (i,*f)).collect();
        indexed_dist.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        //
        // first component store weight cumul , second component stores weight * dist cumul
        //
        let mut cumulated_values = Vec::<(f32,f32)>::with_capacity(self.get_nb_data());
        cumulated_values.push((self.weights[0], indexed_dist[0].1 * self.weights[indexed_dist[0].0]));
        for j in 1..self.get_nb_data() {
            let weight = cumulated_values[j-1].0 + self.weights[indexed_dist[j-1].0];
            let indexed_weight = cumulated_values[j-1].1 + indexed_dist[j].1 * self.weights[indexed_dist[j].0];
            cumulated_values.push((weight,indexed_weight));
        }
        // compute value at ball centered a ball with radius indexed_dist[j].1 (see paper)
        let value_at_j = |j : usize | -> f32 {
            indexed_dist[j].1 * cumulated_values[j].0 - cumulated_values[j].1
        };
        // now we must find greatest j such that value_atj(j) <= value
        // check last value
        let upper_value = value_at_j(self.get_nb_data());
        let radius : f32;
        if upper_value <= value {
            log::info!("value is too large upper_value : {:.3e}", upper_value);
            // we can solve for a large r directly
            radius =  value - upper_value;
            std::panic!("not yet implemented");
        }
        else {
            let mut upper_index = self.get_nb_data() - 1;
            let mut lower_index = 0;
            let mut middle_index = (upper_index + lower_index) / 2;
            let mut value_at_upper = 0.;
            let mut value_at_lower = 0.;
            while upper_index - lower_index > 1 {
                let value_at_middle = value_at_j(middle_index);
                if value_at_middle > value {
                    upper_index = middle_index;
                    value_at_upper = value_at_middle;
                }
                else {
                    lower_index = middle_index;
                    value_at_lower = value_at_middle;
                }
                log::debug!("upper_index - lower_index : {:?}", upper_index - lower_index);
            } // end while
            radius = (value + cumulated_values[lower_index].1)/ cumulated_values[lower_index].0;
            assert!(radius >= indexed_dist[lower_index].1);
            assert!(radius < indexed_dist[upper_index].1);
        } 
        //
        if radius > 0. {
            return radius;
        }
        else {
            std::panic!("error in compute_ball_radius");
        }
    } // end of compute_ball_radius


    //
    // for each point compute ball around it of given value
    // corresponds to step 1 of algorithm 2.1 paper [online-median](https://epubs.siam.org/doi/10.1137/S0097539701383443)
    //
    fn compute_value_balls(&self, value : f32, dists : &Vec<RwLock<Vec<f32>>>) {
        let radii : Vec<f32> = (0..self.nb_data).into_iter().map(|i| self.compute_ball_radius( i, value, &dists[i])).collect();
    }

    // split in blocks of roughly equal weights and make an unweighted  MettuPlaxton problem for each
    // then we can run a weighted algorithm on the union of results.
    fn weightsplit(&self) -> Result<Vec<MettuPlaxton<'b, T, Dist>> > {

        return Err(anyhow!("not yet implemented"));
    } // end of  weightsplit



} // end of impl WeightedMettuPlaxton