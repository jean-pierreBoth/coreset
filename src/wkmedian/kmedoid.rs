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

use hnsw_rs::dist::*;

use crate::sensitivity::*;

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
    // rank of medoids centers in arrays ids and matrix
    centers : Vec<u32>,
    // current affectation of each coreset point
    membership : Vec<u32>,
    // cost by medoid
    costs : Vec<f32>
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
        let centers = (0..nb_cluster).into_iter().map(|_| u32::MAX).collect();
        let membership= (0..nbpoints).into_iter().map(|_| u32::MAX).collect();
        let costs = (0..nb_cluster).into_iter().map(|_| f32::MAX).collect();
        //
        return Kmedoid{nb_cluster, ids, distance, weights, centers, membership, costs}
    } // end of new 


    pub fn compute_medians(&mut self) {
        // init
        self.random_init();
        // iterate
    } // end of compute_medians


    pub fn get_size(&self) -> usize {
        self.ids.len()
    }

    // random initial choice of medoids
    fn random_init(&mut self) {
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
        self.centers = centers;
    } // end of random_init


    // dispatch data to medoids at miniumm cost. can be //
    fn dispatch_to_medoids(&mut self) {
        //
        for i in 0..self.get_size() {
            let (m, _d) = self.find_medoid_for_i(i);
            self.membership[i] = m;
        }
    } // end of dispatch_to_medoids



    // find medoid for point i
    fn find_medoid_for_i(&self, i : usize) -> (u32, f32) {
        //
        let rowi = self.distance.row(i);
        let mut best_m = 0u32;
        let mut best_dist = rowi[self.centers[0] as usize];
        for m in 1..self.centers.len() {
            if rowi[self.centers[m] as usize] < best_dist {
                best_m = self.centers[m];
                best_dist = rowi[self.centers[m] as usize];
            }
        }
        (best_m, best_dist)
    }

    // we mimic kmean update as described in Park-Jun
    // What is the point in cluster that has minimum distance to others indide cluster m.
    // Returns point index (global range 0..self.nb_point) from which points in that cluster has minimum cost to others
    // Cost for center is defined by : Sum{i in m} w(i) * dist(i,c)
    // TODO: must be called in // , costly
    fn from_membership_to_centers(&self) -> Vec<(usize, f32)> {
        // 
        // each i belongs to a cluster, we update the j term contribution inside the same cluster.
        // at end of loop we have contributions of each term to its cluster
        let mut cost : Vec<f32> = (0..self.distance.nrows()).into_iter().map(|_| 0.).collect();
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



    fn iter(&mut self)  {
        self.dispatch_to_medoids();
        let centers = self.from_membership_to_centers();
        // compute global cost
        let global_cost : f32 = centers.iter().map(|x| x.1).sum();
        // TODO: cost evolution ...iteration control
    }



} // end of impl Kmedoid