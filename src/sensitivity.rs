//! Implementation sensitivity sampling as described in:
//! 
//!   - New Fraweworks for Offline and Streaming Coreset Constructions
//!        Braverman, Feldman, Lang, Statsman 2022
//!        [arxiv-v3](https://arxiv.org/abs/1612.00889)

#![allow(unused)]


use std::collections::HashMap;

use rand_distr::WeightedIndex;

use crate::bmor::*;
use crate::facility::*;

use hnsw_rs::dist::*;


// How do we represent a coreset: For now minimal
// We can have same point many times with different weights
// Do we store Vec<T> ?
pub struct Coreset {
    // first usize is point index in data, second is rank in filed weights.
    w_index : HashMap<usize, usize>,
    //
    weights : Vec<Vec<f64>>,
} // end of Coreset



impl Coreset {

    /// returns number of different points
    pub fn get_size(&self) -> usize {
        self.w_index.len()
    }

    /// returns list of weights of a given point if present in coreset
    pub fn get_weights(&self, ptindex : usize) -> Option<&Vec<f64>> {
        std::panic!("not yet");
    }
} // end of impl Coreset




/// This structure provides Algorithm1 Braverman and al 2022.
/// It relies on bmor algorithm [bmor](super::bmor)
/// The algorithm needs one streaming pass and one sampling pass
pub struct Coreset1<T:Send+Sync+Clone, Dist : Distance<T> + Clone + Sync + Send> {
    /// bmor instance
    bmor : Bmor<T, Dist >, 
    /// proba distribution over points
    point_weights : Option<WeightedIndex<usize>>,
    
} // end of Coreset1



// s estimation 

impl <T:Send+Sync+Clone, Dist> Coreset1<T, Dist> 
            where Dist : Distance<T> + Clone + Sync + Send {
    /// 
    pub fn new(k: usize, nbdata_expected : usize, beta : f64, gamma : f64, distance :  Dist) -> Self {
        let bmor = Bmor::new(k, nbdata_expected, beta, gamma, distance);
        let point_weights : Option<WeightedIndex<usize>> = None;
        //
        std::panic!("not yet");
    } // end of new

    /// treat unweighted data. 
    /// **This method can be called many times in case of data streaming, passing data by blocks**.  
    /// It returns the number of facilities created up to this call.
    pub fn process_data(&mut self, data : &[Vec<T>]) -> anyhow::Result<usize> {
        //
        self.bmor.process_data(&data);
        self.bmor.log();
        //
        std::panic!("not yet");
    } // end of process_data


    /// declare end of streaming data, and construct coreset by sampling
    pub fn end_data(&self, contraction : bool) -> Coreset1<T, Dist> {
        std::panic!("not yet");
    } // 




} // end of impl block
