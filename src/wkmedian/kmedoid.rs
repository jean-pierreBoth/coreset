//! Implementation of Kmeds in 
//!     - Park-Jun Simple and Fast algorithm for k-medoids clustering 2009
//!     with :
//!       - random initialization as explained in Newling-Fleuret in Subquadratic Exact medoid algorithm 2017
//!       - introduction of weights attached to data
//! 
//!     See also Friedmann Hastie Tibshirani, The Elements Of Statistical Learning 2001 (Clustering chapter)
//!
//! 
//! 

#![allow(unused)]
use ndarray::Array2;

use rand::{Rng, thread_rng};
use rand::distributions::{Distribution,Uniform};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;

use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::HashMap;

use rayon::iter::{ParallelIterator, IntoParallelIterator};
use quantiles::ckms::CKMS;     // we could use also greenwald_khanna

use hnsw_rs::dist::*;

use crate::sensitivity::*;

pub struct Medoid {
    /// id as given by coreset
    center_id : usize,
    /// rank of points in ids of points as given in struct Coreset
    center : u32,
    /// cost of thi cluster
    cost : f32,
}

impl Medoid {
    //
    fn default() -> Self {
        Medoid{center_id : usize::MAX, center : u32::MAX, cost : f32::MAX}
    }

    fn new(id : usize, center : u32, cost : f32) -> Self {
        Medoid{center_id : id, center, cost}
    }

    pub fn get_center_id(&self) -> usize {
        self.center_id
    }

    fn get_center(&self) -> u32 {
        self.center
    }

    pub fn get_cost(&self) -> f32 {
        self.cost
    }

    fn set_cost(&mut self, cost : f32) {
        self.cost = cost;
    }
}


//TODO: add field from id to rank
/// This algorithm stores the whole matrix distance between points as coreset must have reduced the number of points to a few thousands.
pub struct Kmedoid {
    //
    nb_cluster : usize,
    // orginal ids of data by line of matrix
    ids : Vec<usize>,
    // distance matrix
    distance : Array2<f32>, 
    // weights of points in coreset in order corresponding to lines of distance matrix
    weights : Vec<f32>,
    // current affectation of each coreset point. For each point returns rank in medoids array.
    membership : Vec<u32>,
    // medoid : a Vector containing the nb_cluster medoid
    medoids : Vec<Medoid>,
    //
    dispatching_weights : Vec<f32>
} // end of struct Kmedoid


impl Kmedoid {

    pub fn new<T, Dist>(coreset : &CoreSet<T, Dist>, nb_cluster : usize) -> Self 
            where    T : Send + Sync + Clone,
                  Dist : Distance<T> + Send + Sync + Clone  {
        //
        let (ids, distance) = coreset.compute_distances().unwrap();
        //
        let nbpoints = coreset.get_nb_points();
        log::info!("Kmedoid received coreset of size : {}",nbpoints);
        // weight[i] corresponds to row[i] in distance matrix. From now on all computation use center as raks and no ids.
        let mut weights = Vec::<f32>::with_capacity(nbpoints);
        for i in 0..ids.len() {
            let weight = coreset.get_weight(ids[i]).unwrap();
            weights.push(weight);
        }
        //
        assert_eq!(weights.len(), nbpoints);
        //
        let membership= (0..nbpoints).into_iter().map(|_| u32::MAX).collect();
        let medoids = (0..nb_cluster).into_iter().map(|_| Medoid::default()).collect();
        //
        let dispatching_weights = Vec::<f32>::new();
        //
        return Kmedoid{nb_cluster, ids, distance, weights, membership, medoids, dispatching_weights}
    } // end of new 


    pub fn compute_medians(&mut self) {
        //
        self.quantile_estimator();
        //
        // initialize: random selection of centers, dispatch points to nearest centers
        //
        let mut centers = self.max_dist_init();     // select nb_cluster different points
        //
        self.medoids = centers.iter()
                        .map(|i| Medoid::new(self.ids[*i as usize], *i,f32::MAX))
                        .collect();     
        // dispatch each point to nearest center, i.e set membership.
        let mut centers_and_dist = self.dispatch_to_medoids();
        assert_eq!(self.ids.len(), centers_and_dist.len());
        for i in 0..centers_and_dist.len() {
            self.membership[i] = centers_and_dist[i].0;
        }
        let costs = self.compute_medoids_cost(&centers);   // compute_medoid_cost not called any more after that
        //
        for i in 0..self.medoids.len() {
            self.medoids[i].set_cost(costs[i]);
        }
        // we have centers and cost
        let mut global_cost = costs.iter().sum::<f32>();
        log::info!("medoids initialized , global cost : {:.3e}", global_cost);
        //
        // iterate
        //
        let mut iteration = 0;
        loop {
            // recompute centers from membership and update clusters costs:  Cpu cost is here
            let centers_and_costs = self.from_membership_to_centers();
            // compute global cost
            let mut iter_cost : f32 = centers_and_costs.iter().map(|x| x.1).sum();
            if  iter_cost >= global_cost {
                log::info!("iteration exiting at iteration : {}", iteration);
                break;
            }
            global_cost = iter_cost;
            log::info!("iteration {}, global cost : {:.3e}", iteration, global_cost);
            // we must store our best state
            assert_eq!(centers_and_costs.len(), self.medoids.len() );
            for i in 0..self.medoids.len()  {
                // we do not update center_id we do not use it, we update only at end
                self.medoids[i].center = centers_and_costs[i].0 as u32;
                self.medoids[i].center_id = self.ids[centers_and_costs[i].0];
                self.medoids[i].cost = centers_and_costs[i].1;
            }
            // we must compute distance to new centers and reassign membership
            centers_and_dist  = self.dispatch_to_medoids();
            assert_eq!(centers_and_dist.len(), self.membership.len());
            for i in 0..centers_and_dist.len()  {
                self.membership[i] = centers_and_dist[i].0;
            }
            iteration += 1;
            if iteration >= 1000 {
                log::info!("exiting after nb iteration : {}", iteration);
                break;
            }
        }
        self.quality_summary();
    } // end of compute_medians


    pub  fn get_clusters(&self) -> &Vec<Medoid> {
        &self.medoids
    }


    pub fn get_size(&self) -> usize {
        self.ids.len()
    }

    /// returns for each points the rank of its cluster
    pub fn get_membership(&self) -> &Vec<u32> {
        &self.membership
    }

    /// return global partition cost
    pub fn get_global_cost(&self) -> f32 {
        self.medoids.iter().map(|m| m.get_cost()).sum::<f32>()
    }


    // random initial choice of medoids
    fn random_centers_init(&mut self) -> Vec<u32> {
        // we must iterate until we have k different medoids.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(117);
        let between = Uniform::new::<u32, u32>(0,self.get_size() as u32);
        let mut centers = Vec::<u32>::with_capacity(self.nb_cluster);
        // get k different centers
        while centers.len() < self.nb_cluster {
            let j = between.sample(&mut rng);
            if !centers.contains(&j) {
                centers.push(j);
            }
        }
        centers
    } // end of random_init


    fn max_dist_init(&mut self) ->  Vec<u32> {
        //
        let mut already = vec![false; self.ids.len()];
        let mut centers = Vec::<u32>::with_capacity(self.nb_cluster);
        //
        // We use a weight rescaling to avoid weight being much larger than distances
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(117);
        let between = Uniform::new::<u32, u32>(0,self.get_size() as u32);
        let first = between.sample(&mut rng);
        // choose point of maximal weight
        let mut max_item = (0, self.weights[0]);
        for i in 1..self.weights.len() {
            if self.weights[i] > max_item.1 {
                max_item = (i, self.weights[i]);
            }
        }
        already[max_item.0] = true;
        centers.push(max_item.0 as u32);
        // now search a center for each other cluster
        for c in 1..self.nb_cluster {  
            let dmax : f32 = 0.;
            // search element furthest away from already chosen centers
            let mut cost_item : (usize, f32, f32) = (usize::MAX, -1., -1.);
            for i in 0..self.distance.ncols() {
                if already[i] {
                    continue;
                }
                for c in 0..centers.len() {
                    let dist = self.distance[ [i, centers[c] as usize]];
                    let cost = dist * self.dispatching_weights[i];
                    if cost > cost_item.1 {
                        cost_item = (i, cost, dist);
                    }
                }
            }
            // we furthest point from previous censter, we take it as a new center
            centers.push(cost_item.0 as u32);
            assert!(!already[cost_item.0]);
            already[cost_item.0] = true;
            log::info!("new center; i : {}, cost : {:.3e} dist to previous centers : {:.3e}", cost_item.0, cost_item.1, cost_item.2);
        }
        //
        return centers;
    }

    // dispatch data to medoids. Returns for each data point cluster number and distance to center of the cluster
    fn dispatch_to_medoids(&mut self) -> Vec<(u32, f32)> {
        //
        let centers_dist : Vec<(u32, f32)> = (0..self.get_size()).into_par_iter().map(|i| self.find_medoid_for_i(i)).collect();
        //
        centers_dist
    } // end of dispatch_to_medoids



    // find medoid for point i, returns rank of cluster with nearest center to i
    fn find_medoid_for_i(&self, i : usize) -> (u32, f32) {
        //
        let rowi = self.distance.row(i);
        let mut best_m = 0u32;
        let mut best_dist = rowi[self.medoids[0].get_center() as usize];
        for m in 1..self.medoids.len() {
            let test_m = self.medoids[m].get_center();
            if rowi[test_m as usize] < best_dist {
                // affect to best medoid index
                best_m = m as u32;
                best_dist = rowi[test_m as usize];
            }
        }
        (best_m, best_dist)
    }

    // argument centers is such that centers[k] contains the rank of cluster k 
    // if we know centers and membership we can compute costs of each cluster.
    fn compute_medoids_cost(&self, centers : &Vec<u32>) -> Vec<f32> {
        //
        log::info!("initial medoid affectation");
        // This function computes cost of cluster containing i if i is its center
        let cost_i = | i | -> (usize, f32) {
            let mut cost : f32 = 0.;
            let mut cluster_size : usize = 0;
            let i_cluster = self.membership[i];
            log::debug!("compute_medoid_cost for arg i : {}, center : {} ",i, i_cluster);
            for j in 0..self.get_size() {
                if j != i && self.membership[j] == i_cluster {
                    let delta_c = self.distance[[i,j]] * self.dispatching_weights[j];
                    cluster_size += 1;
                    cost += delta_c;
                    log::trace!("     member  : {}, increment  cost : {:.3e}", j, delta_c);
                }
            }
            (cluster_size, cost)
        };
        // TODO: to be made //
        assert_eq!(centers.len(), self.nb_cluster);
        let mut costs : Vec<f32> = (0..self.nb_cluster).into_iter().map(|_| 0.).collect();
        for i in 0..self.medoids.len() {
            let cluster_size : usize;
            (cluster_size, costs[i]) = cost_i(centers[i] as usize);
            log::info!("medoid i : {}, center : {}, weight : {:.3e} cost : {:.3e} size : {}  \n\n ", i, centers[i], 
                            self.weights[centers[i] as usize ] , costs[i], cluster_size);
        }
        costs
    } // end of compute_initial_cost



    // we mimic kmean update as described in Park-Jun
    // What is the point in cluster that has minimum distance to others indide cluster m.
    // Returns point index (global range 0..self.nb_point) from which points in that cluster has minimum cost to others
    // Cost for center is defined by minimizing cost :  Sum{i in m} w(i) * dist(i,c)
    //
    // This function returns for each cluster a 2-uple containing (center, cluster cost)
    fn from_membership_to_centers(&self) -> Vec<(usize, f32)> {
        // 
        // each point i belongs to a cluster, we update the j term contribution inside the same cluster.
        // at end of loop we have contributions of each term to its cluster
        // this function returns cost of cluster having i as member if we consider i as center
        let cost_i = | i | -> f32 {
            let mut cost : f32 = 0.;
            let i_cluster = self.membership[i];
            for j in 0..self.get_size() {
                // if same medoid, update cost
                if j != i && self.membership[j] == i_cluster {
                    cost += self.distance[[i,j]] * self.dispatching_weights[j];
                }
            }
            cost
        };
        // TODO: iterate and collect!
         
        let mut cost : Vec<f32>;
        if self.get_size() < 10000 {
            cost  = (0..self.distance.nrows()).into_iter().map(|_| 0.).collect();
            for i in 0..self.get_size() {
                cost[i] = cost_i(i);
            }
        }
        else {
            cost = (0..self.distance.nrows()).into_par_iter().map(|i| cost_i(i)).collect();
        }
        //
        let mut centers : Vec<(usize, f32)> = (0..self.nb_cluster).into_iter().map(|_| (usize::MAX, f32::MAX)).collect();
        for i in 0..self.get_size() {
            let c = self.membership[i];
            assert!( (c as usize) < centers.len());
            if cost[i] < centers[c as usize].1 {
                centers[c as usize].1 = cost[i];
                centers[c as usize].0 = i;
            }
        }
        // 
        centers
    } // end of from_membership_to_centers


    // computes statistics (quantiles) of distances to their centers
    fn quality_summary(&self) {
        log::info!("\n\n kmedoids statistics");
        //
        let mut q_dist = CKMS::<f32>::new(0.01);
        for i in 0..self.distance.nrows() {
            let m = self.membership[i];
            let c = self.medoids[m as usize].center as usize;
            q_dist.insert(self.distance[[i,c]]);
        }
        println!("\n distance to centroid quantiles at 0.01 :  {:.2e} , 0.025 : {:.2e}, 0.5 : {:.2e}, 0.75 : {:.2e}   0.99 : {:.2e}\n", 
            q_dist.query(0.01).unwrap().1,  q_dist.query(0.025).unwrap().1, 
            q_dist.query(0.5).unwrap().1, q_dist.query(0.75).unwrap().1, q_dist.query(0.95).unwrap().1);
        //
        let mut medoids_size = vec![0u32; self.nb_cluster];
        let mut medoids_dist_sum = vec![0.0f32; self.nb_cluster];
        for i in 0..self.membership.len() {
            let m = self.membership[i] as usize;
            medoids_size[m] += 1;
            medoids_dist_sum[m] += self.distance[[i, self.medoids[m].center as usize]];
        }
        for m in 0..self.medoids.len() {
            log::info!("medoid : {}, size : {} , cost {:.2e} , mean dist : {:.2e}", 
                        m ,medoids_size[m], self.medoids[m].cost, medoids_dist_sum[m]/ medoids_size[m] as f32)
        }
    } // end of quality_summary



    /// estimate quantiles of distances
    pub fn quantile_estimator(&mut self) ->  CKMS::<f32> {

        log::info!("statistics on weights");
        let mut quantiles = CKMS::<f32>::new(0.01);
        for w in &self.weights {
            quantiles.insert(*w);
        }
        println!("\n weights quantiles at  0.01 :  {:.2e} , 0.025 : {:.2e}, 0.05 : {:.2e}, 0.5 : {:.2e}, 0.75 : {:.2e}   0.99 : {:.2e}\n", 
                quantiles.query(0.01).unwrap().1,  quantiles.query(0.025).unwrap().1,  quantiles.query(0.05).unwrap().1,
                quantiles.query(0.5).unwrap().1, quantiles.query(0.75).unwrap().1, quantiles.query(0.99).unwrap().1);
        
        let wlow = quantiles.query(0.25).unwrap().1;
        let wupper = quantiles.query(0.75).unwrap().1;
        let wmedian = quantiles.query(0.5).unwrap().1;

        self.dispatching_weights.clear();
        for i in 0..self.weights.len() {
//            self.dispatching_weights.push(self.weights[i].max(wlow).min(wupper) / wmedian);
            self.dispatching_weights.push(self.weights[i]);
        }
        //
        log::info!("statistics on distances");
        let nbrow = self.distance.nrows();
        let nbcol = self.distance.ncols();
        //
        let mut quantiles  = CKMS::<f32>::new(0.01);
        for i in 0..nbrow {
            for j in 0..nbcol {
                    if i != j {
                        quantiles.insert(self.distance[[i,j]]);
                    }
            }
        }
        println!("\n distance quantiles at 0.0001 : {:.2e} , 0.001 : {:.2e}, 0.01 :  {:.2e} , 0.025 : {:.2e}, 0.05 : {:.2e},, 0.5 : {:.2e}, 0.75 : {:.2e}   0.99 : {:.2e}\n", 
            quantiles.query(0.0001).unwrap().1, quantiles.query(0.001).unwrap().1,  quantiles.query(0.01).unwrap().1,  quantiles.query(0.025).unwrap().1, 
            quantiles.query(0.05).unwrap().1, quantiles.query(0.5).unwrap().1, quantiles.query(0.75).unwrap().1, quantiles.query(0.99).unwrap().1);

        return quantiles;  
    }




} // end of impl Kmedoid



