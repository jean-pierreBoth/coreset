//!  This module implement building blocks used a black box in coreset constructions.
//!  The algorithms compute an (alfa, beta) k-median approximation  used as input
//!  to coreset computations.
//! 
//! The module implements of the Mettu-Plaxton algorithm as analyzed in 
//!  1. Facility Location in sublinear time.   
//!       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
//!       see [Badoiu](https://people.csail.mit.edu/indyk/fl.pdf).  
//!       This paper builds upon the following: 
//!        
//! 
//! 
//!  2. The online median problem,  
//!         Mettu-Plaxton Siam 2003 [online-median](https://epubs.siam.org/doi/10.1137/S0097539701383443).  
//! 
//! 
//!  3. Optimal Time bounds for approximate Clustering   
//!     Mettu-Plaxton 2004.  see [mettu-plaxton-2](https://link.springer.com/article/10.1023/b:mach.0000033114.18632.e0).  
//! 
//!  The data we will run on  Vec\<T\> where T can be anything as long as the hnsw crate provides on these vectors.  
//!  (see [hnsw_rs](https://docs.rs/hnsw_rs/0.1.19/hnsw_rs/dist/index.html))
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


/// Mettu-Plaxton algorithm with Indyk simplification as described in  [indyk](https://people.csail.mit.edu/indyk/fl.pdf).  
/// This algorithm suppose uniform weights on data.  
/// 
/// If data have weights attached, data can be split in subsets of approximately equal weights as in
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
    pub fn construct_centers(&self, alfa : f32) -> Facilities<T, Dist>
         where Dist : Send + Sync + Clone {
        // get scales
        let q_dist = get_neighborhood_size(1_000_000, self.data, &self.distance);
        let d_range = (q_dist.query(0.0001).unwrap().1, q_dist.query(0.95).unwrap().1);
        let threshold = q_dist.query(0.999).unwrap().1;
        log::info!("dist medi : {:.3e}",threshold);
        //
        let mut facilities: Facilities<T, Dist> = Facilities::<T, Dist>::new(self.j as usize, self.distance.clone());
        // estimate radii in //
        let value_to_match = alfa * threshold;
        let mut radii : Vec<(usize,f32)> = (0..self.nb_data).into_par_iter().map(|i| self.estimate_ball_cardinal((i, &self.data[i]), value_to_match)).collect();
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


/// Mettu-Plaxton online median algorithm Siam 2003 [online-median](https://epubs.siam.org/doi/10.1137/S0097539701383443).  
/// This algorithm can handle weighted data but its complexity is O(n²).  
/// 
/// It is adapted to final processing of Coreset algorithms where number of items to process has been log reduced (bmor module algos   
/// or blackbox as in Chen K., On Coresets Kmedian Clustering Metric Spaces and Applications 2009 Siam J. Computing)
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


impl <'b, T:Send+Sync+Clone, Dist : Distance<T> + Send + Sync + Clone> WeightedMettuPlaxton<'b,T, Dist> {

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
        log::debug!("in WeightedMettuPlaxton::compute_all_dists");
        //
        let mut dists : Vec<RwLock<Vec<f32>>> = Vec::<RwLock<Vec<f32>>>::with_capacity(self.nb_data);
        //
        for i in 0..self.nb_data {
            let d :Vec<f32> = (0..self.nb_data).into_iter().map(|_| -1.).collect();
            dists.push(RwLock::new(d));
        }

        let threshold = 1000;
        let compute_for_i = | i : usize |  {
            let mut dist_i : Vec<f32> = (0..self.nb_data).into_iter().map(|_| -1.).collect();
            for j in 0..self.nb_data {
                // has symetric benn computed?
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


    fn compute_ball_radius(&self, ball : usize, alfa : f32, dists  : &RwLock<Vec<f32>>) -> f32 {
        //
        log::debug!("\n\n WeightedMettuPlaxton compute_ball_radius , coeff value to match {:.3e}", alfa);
        log::debug!("WeightedMettuPlaxton compute_ball_radius , radii  {:?}", dists.read());
        //
        // we sort distances
        let mut indexed_dist : Vec<(usize, f32)> = (0..self.nb_data).into_iter().zip(dists.read().iter()).map(|(i, f)| (i,*f)).collect();
        indexed_dist.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        log::debug!("indexed_dist : {:?}",indexed_dist);
        //
        // first component store weight cumul , second component stores weight * dist cumul
        //
        let mut cumulated_values = Vec::<(f32,f32)>::with_capacity(self.get_nb_data());
        cumulated_values.push((self.weights[indexed_dist[0].0], indexed_dist[0].1 * self.weights[indexed_dist[0].0]));
        for j in 1..self.get_nb_data() {
            let weight = cumulated_values[j-1].0 + self.weights[indexed_dist[j].0];
            let indexed_weight = cumulated_values[j-1].1 + indexed_dist[j].1 * self.weights[indexed_dist[j].0];
            cumulated_values.push((weight,indexed_weight));
        }
        assert_eq!(cumulated_values.len(), self.get_nb_data());
        log::debug!("cumulated_values :  {:?}" , cumulated_values);
        // compute value at ball centered a ball with radius indexed_dist[j].1 (see paper)
        let value_at_j = |j : usize | -> f32 {
            indexed_dist[j].1 * cumulated_values[j].0 - cumulated_values[j].1
        };
        //
        let radius_index = self.nb_data.ilog2().max(1);
        let value = alfa * value_at_j(radius_index.try_into().unwrap());
        log::debug!("value to match : {:.2e}", value);
        if log::log_enabled!(log::Level::Debug) {
            let check : Vec<f32> = (0..self.get_nb_data()).into_iter().map(|j| value_at_j(j)).collect();
            log::debug!("check : {:?}", check);
        }
        //
        // now we must find greatest j such that value_atj(j) <= value
        // check last value
        let upper_value = value_at_j(self.get_nb_data() - 1);
        log::debug!("imp::compute_ball_radius upper_value : {:.3e} value : {:.3e}", upper_value, value);
        let radius : f32;
        if upper_value <= value {
            log::info!("value is too large upper_value : {:.3e} value : {:.3e}", upper_value, value);
            // we can solve for a large r directly
            radius =  value - upper_value;
            std::panic!("not yet implemented");
        }
        else {
            let mut upper_index = self.get_nb_data() - 1;
            let mut lower_index = 0;
            let mut middle_index = (upper_index + lower_index) / 2;
            let mut value_at_upper = upper_value;
            let mut value_at_lower = 0.;
            while upper_index - lower_index > 1 {
                log::trace!("lower index: {:?}, upper index : {:?} value lower : {:?}, value upper {:?}", lower_index, upper_index, value_at_lower, value_at_upper);
                let value_at_middle = value_at_j(middle_index);
                if value_at_middle > value {
                    upper_index = middle_index;
                    value_at_upper = value_at_middle;
                }
                else {
                    lower_index = middle_index;
                    value_at_lower = value_at_middle;
                }
                middle_index = (upper_index + lower_index) / 2;
            } // end while
            log::debug!("lower index : {:?}, upper index : {:?}, value lower  : {:?}, value upper : {:?} ", lower_index, upper_index,value_at_lower, value_at_upper);
            radius = indexed_dist[lower_index].1 + (value - value_at_lower) / cumulated_values[lower_index].0;
            log::debug!("got radius : {:?}", radius);
            assert!(radius >= indexed_dist[lower_index].1);
            assert!(radius < indexed_dist[upper_index].1);
        } 
        //
        if radius > 0. {
            return radius;
        }
        else {
            std::panic!("error in compute_ball_radius, radius : {:?}", radius);
        }
    } // end of compute_ball_radius



    //
    // TODO: to made //
    fn compute_balls_at_value(&self, value : f32, dists : &Vec<RwLock<Vec<f32>>>) -> Facilities<T, Dist> {
        //
        //
        log::debug!("in WeightedMettuPlaxton::compute_balls_at_value");
        // for each point compute ball around it of given value
        // corresponds to step 1 of algorithm 2.1 paper [online-median](https://epubs.siam.org/doi/10.1137/S0097539701383443)        
        let mut radii : Vec<(usize, f32)> = (0..self.nb_data).into_iter().map(|i| (i, self.compute_ball_radius( i, value, &dists[i]))).collect();
        // sort by increasing radius (step 2 of algo)
        radii.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        // radii[4] corresponds to point of original index radii[4].0 and so distance to its neighbours are given by dists[radii[4].0]
        let mut facilities: Facilities<T, Dist> = Facilities::<T, Dist>::new(self.j as usize, self.distance.clone());
        // we can do step 3 and 4 of algo 2.1
        for p in radii.iter() {
            let matched = facilities.match_point(&self.data[p.0],  2. * p.1, &self.distance);
            if !matched {
                // we insert a facility
                let mut facility = Facility::new(p.0, &self.data[p.0]);
                facility.insert(self.weights[p.0] as f64, p.1 as f32);
                log::info!("inserting facility at {:?}, radius : {:.3e}, weight : {:.3e}", p.0, p.1, facility.get_weight());
                facilities.insert(facility);
            }
        }
        return facilities;
    } // end of compute_balls_at_value


    /// alfa governs the number of facilities we will get.
    /// alfa = 0.5 is a good value. To reduce number of facilities produced decrease alfa.
    pub fn construct_centers(&self, alfa : f32) -> Facilities<T, Dist> {
        //
        log::debug!("in WeightedMettuPlaxton::construct_centers");
        //
        let dists : Vec<RwLock<Vec<f32>>> = self.compute_all_dists();
        //
        let facilities = self.compute_balls_at_value(alfa, &dists);
        //
        return facilities;
    } // end of construct_centers





} // end of impl WeightedMettuPlaxton



mod tests {

    use super::*;
    use rand::prelude::*;
    use rand::distributions::*;
    use rand_distr::{Normal, uniform::SampleUniform};
    use rand_xoshiro::Xoshiro256PlusPlus;



    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // generate data according to 2 gaussian distributions and random uniform weights sampled in intervals.
    fn generate_weighted_data(nbdata : usize) -> (Vec::<Vec<f32>>, Vec<f32>) {
        //
        let dim = 50;
        let mut data = Vec::<Vec<f32>>::with_capacity(nbdata);
        let mut weights = Vec::<f32>::with_capacity(nbdata);
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1454691);
        //
        // distributions
        //
        let n_mean1 = 2.;
        let n_sigma1 = 1.;
        let normal1 = Normal::new(n_mean1, n_sigma1).unwrap();
        let unif1 = Uniform::<f32>::new(0.5, 3.);
        //
        let n_mean2 = 4.;
        let n_sigma2 = 1.;
        let normal2 = Normal::new(n_mean1, n_sigma1).unwrap();
        let unif2 = Uniform::<f32>::new(0.5, 3.);  
        //
        // sample
        //      
        let half = nbdata/2;
        for i in 0..half {
            let d_tmp : Vec<f32> = (0..dim).into_iter().map(|_| normal1.sample(&mut rng)).collect();
            data.push(d_tmp);
            weights.push(unif1.sample(&mut rng));
        }

        for i in (half+1)..nbdata {
            let d_tmp : Vec<f32> = (0..dim).into_iter().map(|_| normal2.sample(&mut rng)).collect();
            data.push(d_tmp);
            weights.push(unif2.sample(&mut rng));
        }
        // log::debug!("data : {:?}", data);
        // log::debug!("weights : {:?}", weights);
        //
        return (data, weights);
    } // end of generate_weighted_data

#[test]
    fn test_weight_mp() {
        log_init_test();
        //
        log::info!("in test_weight_mp");
        //
        let (data, weights) = generate_weighted_data(12);
        let distance = DistL2::default();
        //
        let wmp = WeightedMettuPlaxton::new(&data, &weights, distance);
        //
        let alfa = 1.;
        let facilities = wmp.construct_centers(alfa);
    } // end of test_weight_mp


}  // end of mod tests

