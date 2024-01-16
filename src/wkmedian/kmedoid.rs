//! Implementation of Kmeds in 
//!     - Park-Jun Simple and Fast algorithm for k-medois clustering 2009
//!     with :
//!       - random initialization as explained in Newling-Fleuret in Subquadratic Exact medoid algorithm 2017
//!       - introduction of weights attached to data
//! 
//!     See also Friedmann Hastie Tibshirani, The Elements Of Statistical Learning 2001
//!
//! 
//! 

#![allow(unused)]
use ndarray::Array2;

use hnsw_rs::dist::*;

use crate::sensitivity::*;

/// This algorithm stores the whole matrix distance between points as coreset must have reduced the number of points to a few thousands.
pub struct Kmedoid {
    // orginal ids of data by line of matrix
    ids : Vec<usize>,
    // distance matrix
    distance : Array2<f32>, 
    // weights of points in coreset in order corresponding to lines of distance matrix
    weights : Vec<f32>
} // end of struct Kmedoid


impl Kmedoid {

    pub fn new<T, Dist>(coreset : &CoreSet<T, Dist>) -> Self 
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
        return Kmedoid{ids, distance, weights}
    } // end of new 





} // end of impl Kmedoid