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

use rayon::iter::{ParallelIterator, IntoParallelIterator};

use hnsw_rs::dist::*;

use crate::sensitivity::*;

struct Medoid {
    center : u32,
    cost : f32,
}

impl Medoid {
    //
    fn default() -> Self {
        Medoid{center : u32::MAX, cost : f32::MAX}
    }

    fn new(center : u32, cost : f32) -> Self {
        Medoid{center, cost}
    }

    fn get_center(&self) -> u32 {
        self.center
    }

    fn get_cost(&self) -> f32 {
        self.cost
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
    // current affectation of each coreset point
    membership : Vec<u32>,
    // medoid
    medoids : Vec<Medoid>
} // end of struct Kmedoid


impl Kmedoid {

    pub fn new<T, Dist>(coreset : &CoreSet<T, Dist>, nb_cluster : usize) -> Self 
            where    T : Send + Sync + Clone,
                  Dist : Distance<T> + Send + Sync + Clone  {
        //
        let (ids, distance) = coreset.compute_distances().unwrap();
        //
        let nbpoints = coreset.get_nb_points();
        let mut weights = Vec::<f32>::with_capacity(nbpoints);
        for id in &ids {
            let weight = coreset.get_weight(*id).unwrap();
            weights.push(weight);
        }
        //
        let membership= (0..nbpoints).into_iter().map(|_| u32::MAX).collect();
        let medoids = (0..nb_cluster).into_iter().map(|_| Medoid::default()).collect();
        //
        return Kmedoid{nb_cluster, ids, distance, weights, membership, medoids}
    } // end of new 


    pub fn compute_medians(&mut self) {
        //
        // initialize: random selection of centers, dispatch points to nearest centers
        //
        let mut centers = self.random_centers_init();
        // dispatch to nearest center, i.e set membership
        let mut centers_and_dist = self.dispatch_to_medoids();
        //
        let costs = self.compute_medoid_cost(&centers);
        // we have centers and cost
        let medoids : Vec<Medoid> = centers.iter().zip(costs.iter()).map(|(i,f)| Medoid::new(*i,*f)).collect();
        let initial_cost = costs.iter().sum::<f32>();
        log::info!("medoids initialized , global cost : {:.3e}", initial_cost);
        //
        // iterate
        //
        let mut iteration = 0;
        loop {
            // recompute centers from membership and update clusters costs:  Cpu cost is here
            let centers_and_costs = self.from_membership_to_centers();
            // compute global cost
            let global_cost : f32 = centers_and_costs.iter().map(|x| x.1).sum();
            if global_cost > initial_cost {
                break;
            }
            log::info!("iteration {}, global cost : {:.3e}", iteration, global_cost);
            // we must store our best state
            assert_eq!(centers_and_costs.len(),medoids.len() );
            for i in 0..medoids.len()  {
                self.medoids[i].center = centers_and_costs[i].0 as u32;
                self.medoids[i].cost = centers_and_costs[i].1;
            }
            // we must compute distance to new centers and reassign membership
            centers_and_dist  = self.dispatch_to_medoids();
            assert_eq!(centers_and_dist.len(), self.membership.len());
            for i in 0..centers_and_dist.len()  {
                self.membership[i] = centers_and_dist[i].0;
            }
        }
    } // end of compute_medians


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


    // dispatch data to medoids. Returns for each point cluster number and distance to center of medoid
    fn dispatch_to_medoids(&mut self) -> Vec<(u32, f32)> {
        //
        let centers_dist : Vec<(u32, f32)> = (0..self.get_size()).into_par_iter().map(|i| self.find_medoid_for_i(i)).collect();
        //
        centers_dist
    } // end of dispatch_to_medoids



    // find medoid for point i
    fn find_medoid_for_i(&self, i : usize) -> (u32, f32) {
        //
        let rowi = self.distance.row(i);
        let mut best_m = 0u32;
        let mut best_dist = rowi[self.medoids[0].get_center() as usize];
        for m in 1..self.medoids.len() {
            let test_m = self.medoids[m].get_center();
            if rowi[test_m as usize] < best_dist {
                best_m = test_m;
                best_dist = rowi[test_m as usize];
            }
        }
        (best_m, best_dist)
    }


    // if we know centers and membership we can compute costs.
    fn compute_medoid_cost(&self, centers : &Vec<u32>) -> Vec<f32>{
        //
        let cost_i = | i | -> f32 {
            let mut cost : f32 = 0.;
            let i_cluster = self.membership[i];
            for j in 0..self.get_size() {
                if j != i && self.membership[j] == i_cluster {
                    cost += self.distance[[i,j]] * self.weights[j];
                }
            }
            cost
        };
        // TODO: to be made //
        let mut costs : Vec<f32> = (0..self.nb_cluster).into_iter().map(|_| 0.).collect();
        for i in 0..self.medoids.len() {
            costs[i] = cost_i(centers[i] as usize)
        }
        costs
    } // end of compute_medoid_cost



    // we mimic kmean update as described in Park-Jun
    // What is the point in cluster that has minimum distance to others indide cluster m.
    // Returns point index (global range 0..self.nb_point) from which points in that cluster has minimum cost to others
    // Cost for center is defined by minimizing cost :  Sum{i in m} w(i) * dist(i,c)
    //
    // This function returns for each cluster a 2-uple containing (center, cluster cost)
    // TODO: must be called in // , costly
    fn from_membership_to_centers(&self) -> Vec<(usize, f32)> {
        // 
        // each point i belongs to a cluster, we update the j term contribution inside the same cluster.
        // at end of loop we have contributions of each term to its cluster
        let mut cost : Vec<f32> = (0..self.distance.nrows()).into_iter().map(|_| 0.).collect();
        // this function returns cost of cluster having i as member if we consider i as center
        let cost_i = | i | -> f32 {
            let mut cost : f32 = 0.;
            let i_cluster = self.membership[i];
            for j in 0..self.get_size() {
                if j != i && self.membership[j] == i_cluster {
                    cost += self.distance[[i,j]] * self.weights[j];
                }
            }
            cost
        };
        // TODO: iterate and collect!
        for i in 0..self.get_size() {
            cost[i] = cost_i(i);
        }
        //
        let mut centers : Vec<(usize, f32)> = (0..self.nb_cluster).into_iter().map(|_| (usize::MAX, f32::MAX)).collect();
        for i in 0..self.get_size() {
            let c = self.membership[i];
            if cost[i] < centers[c as usize].1 {
                centers[c as usize].1 = cost[i];
                centers[c as usize].0 = i;
            }
        }
        // 
        centers
    } // end of from_membership_to_centers




} // end of impl Kmedoid