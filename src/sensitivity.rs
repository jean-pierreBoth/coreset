//! Implementation sensitivity sampling as described in:
//! 
//!   - New Fraweworks for Offline and Streaming Coreset Constructions
//!        Braverman, Feldman, Lang, Statsman 2022
//!        [arxiv-v3](https://arxiv.org/abs/1612.00889)

// We need 2 passes on data as Bmor algorithm can merge data when rescaling cost and facility number so data id are not conserved.

#![allow(unused)]

use anyhow::*;

use std::sync::Arc;
use rayon::prelude::*;

use std::collections::HashMap;
use std::collections::hash_map;
use dashmap::DashMap;

use rand::Rng;
use rand_distr::{Distribution,WeightedIndex};


use crate::bmor::*;
use crate::facility::*;

use hnsw_rs::dist::*;


struct PointSampler {
    /// proba distribution over points
    point_weights : WeightedIndex<f32>,
    // first usize is an index in point_weights, second index is data_id in data (possibly an index).
    w_index : HashMap<usize, usize>,
}

impl PointSampler {

    fn new(weights : Vec<f32>, w_index : HashMap<usize, usize>) -> Self {
        PointSampler{point_weights : WeightedIndex::new(&weights).unwrap() , w_index}
    }

    /// sample a random data point, returning its Id (possibly an index in some array)
    fn sample<R>(&self, rng : &mut R ) -> usize 
        where R : Rng + ?Sized  {
            let rank = self.point_weights.sample(rng);
            return *self.w_index.get(&rank).unwrap();
    } // end of sample


} // end of PointSampler


//======================================================================================================

// How do we represent a coreset: For now minimal
// We can have same point many times with different weights
// Do we store Vec<T> ?
pub struct Coreset {
    // first usize is a data_id in data (possibly an index), second is rank in field weights.
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
    pub fn get_weights(&self, data_id : usize) -> Option<&Vec<f64>> {
        let index_res = self.w_index.get(&data_id);
        match index_res {
            Some(index) => {
                return Some(&self.weights[*index]);
            }
            _ => { return None; }
        }
    } // end of get_weights


    pub fn get_data_ids(&self) -> hash_map::Keys<usize, usize> { return self.w_index.keys()}
} // end of impl Coreset




/// This structure provides Algorithm1 Braverman and al 2022.
/// It relies on bmor algorithm [bmor](super::bmor)
/// The algorithm needs  one streaming pass and one sampling pass.
/// The data must be given consistent id across the 2 passes. (The data id can be its rank in the stream in which case the 2 pass
/// must process data in the same order)
pub struct Coreset1<T:Send+Sync+Clone, Dist : Distance<T> + Clone + Sync + Send> {
    ///
    phase : usize,
    /// keep track of number of data processed
    nb_data : usize,
    /// bmor instance
    bmor : Bmor<T, Dist >, 
    /// facilities with respect to which we compute sensitivity (or importance)
    facilities : Option<Facilities<T, Dist>>,
    /// coreset. A data point can occur many times with varying (decresing weights)
    coreset : HashMap<usize, Vec<f32>>,
    // A map to store facility associated to each point. Needed in sensitivity
    p_facility_map : Option<Arc<DashMap<usize,PointMap>>>,
} // end of Coreset1



// s estimation 

impl <T:Send+Sync+Clone, Dist> Coreset1<T, Dist> 
            where Dist : Distance<T> + Clone + Sync + Send {
    /// 
    pub fn new(k: usize, nbdata_expected : usize, beta : f64, gamma : f64, distance :  Dist) -> Self {
        let bmor = Bmor::new(k, nbdata_expected, beta, gamma, distance);
        let phase = 0usize;
        let coreset = HashMap::<usize, Vec<f32>>::new();
        //
        Coreset1{ phase, nb_data : 0, bmor, facilities : None, coreset, p_facility_map : None}
    } // end of new


    pub fn get_coreset(&self) -> &HashMap<usize, Vec<f32>> {
        panic!("not yet");
    }


    /// treat unweighted data. 
    /// **This method can be called many times in case of data streaming, passing data by blocks**.  
    /// At end of first round on data (end_pass)[end_pass] must be called before running the second pass on data
    pub fn process_data(&mut self, data : &[Vec<T>], data_id : &[usize]) -> anyhow::Result<()> {
        //
        if self.phase == 0 {
            self.bmor.process_data(&data, data_id);
            self.bmor.log();
            self.nb_data += data.len();
            return Ok(());
        }
        else {
            let facilities_ref = self.facilities.as_ref().unwrap();
            let f_map = self.get_facility_map().unwrap();
            //
            let dispatch_i = | item : usize | {
                // get facility rank and weight
                let (facility, dist) = facilities_ref.get_nearest_facility(&data[item], false).unwrap();           
                match &self.p_facility_map {
                    Some(f_map) => {
                        let p_map = PointMap::new(facility, dist, 1.);
                        let res = f_map.insert(data_id[item], p_map);
                        if res.is_some() {
                            log::error!("data_id {} is already present error", data_id[item]);
                            std::panic!();
                        }
                    }
                    _  => { std::panic!("should not happen") }
                }          
            };
            // now we insert into pointmap if necessary
            (0..data.len()).into_par_iter().for_each( |item| dispatch_i(data_id[item]));
            //
            return Ok(());
        };
     } // end of process_data





    /// declare end of streaming data first pass, and construct coreset by sampling
    pub fn end_pass(&mut self) -> Coreset1<T, Dist> {
        //
        match self.phase {
            0 => {
                // we retrieve facilities
                self.phase += 1;
                // we have facilities
                let contraction = false;
                self.facilities = Some(self.bmor.end_data(contraction));
                log::info!("end of first pass, processed nb data : {:?}", self.nb_data);
            }

            1 => {
                // we have every thing to compute sensitivity and do sampling
                log::info!("end of second pass, doing sensitivity and sampling computations");
                self.init_facility_map(self.nb_data);
                // build sampling distribution
//                let sampler = self.build_sampling_distribution(dataiter)
            }

            _ => {
                std::panic!("should not occurr");
            }

        }
        // now we have cardinal, weight and cost for all facilities
        std::panic!("not yet");
    } // 


    /// returns a reference to (optional) facility_map
    pub fn get_facility_map(&self) -> Option<&Arc<DashMap<usize, PointMap>>> {
        return self.p_facility_map.as_ref()
    }

    // if needed (as in the case of sensitivity computations) allocate 
    fn init_facility_map(&mut self, capacity : usize) {
            self.p_facility_map = Some(Arc::new(DashMap::with_capacity(capacity)))
    }

    // if there is a p_facility_map we store pointmap
    fn insert_pointmap(&self, id : usize, pointmap : PointMap) -> anyhow::Result<()> {
        let res = match &self.p_facility_map {
            Some(f_map) => {
                let insert_res = f_map.insert(id, pointmap);
                if insert_res.is_some() {
                    Err(anyhow!("data id not unique id is : {}", id))
                }
                else {
                    Ok(())
                }
            }
            _  => { std::panic!("should not occur, p_facility_map not allocated")}
        };
        //
        return res;
    } // end of insert_pointmap


    //  initialize point_weights : Option<WeightedIndex<usize>>,
    // The function receive an iterator over data.
    // from rust 1.75 such an iterator can be obtained with the user implementing a trait providing the iterator on data
    fn build_sampling_distribution(&mut self, mut dataiter : impl Iterator<Item=(usize, Vec<T>)> ) -> PointSampler {
        // The 2 denonimators used in line 3 of algo 1 for Coreset in Braverman
        let facililities_ref = self.facilities.as_ref().unwrap();
        // denominator used in line 3  of algo 1 for Coreset in Braverman
        let global_cost = facililities_ref.get_cost();
        let p_facility_map_ref = self.p_facility_map.as_ref().unwrap();
        let nb_facilities = facililities_ref.len();     // This is |B| in line 3  of algo 1 for Coreset in Braverman
        let mut cumul_proba = 0.;
        // the fields to build PointSampler
        let mut p_weights = Vec::<f32>::with_capacity(self.nb_data);
        let mut w_index = HashMap::<usize, usize>::with_capacity(self.nb_data);
        //
        while let Some(dataterm) = dataiter.next() {
            let dataid = dataterm.0;
            let data = dataterm.1;
            let pointmap = p_facility_map_ref.get(&dataid);
            if pointmap.is_none() {
                log::error!("error, dataid not found in p_facility_map : {}", dataid);
                std::panic!("error, dataid not found in p_facility_map : {}", dataid);
            }
            let pointmap = pointmap.unwrap();
            let mut proba = pointmap.get_dist() as f64 * pointmap.get_weight() as f64 / global_cost;
            // get weight of facility of point corresponding to data_id
            let f_weight = facililities_ref.get_facility_weight(pointmap.get_facility());
            proba = proba + pointmap.get_weight() as f64 / (nb_facilities as f64 *  f_weight.unwrap());
            proba = proba * 0.5;
            // now we can update 
            assert!(proba > 0.);
            p_weights.push(proba as f32);
            w_index.insert(dataid, p_weights.len());
            // for a check
            cumul_proba += proba;
        }
        assert!((1. - cumul_proba).abs() < 1.0E-5);
        let point_sampler = PointSampler::new(p_weights, w_index);
        // we can now get rid of p_facility_map
        self.p_facility_map = None;
        //
        return point_sampler;
    } // end of build_sampling_distribution



    // build and init field coreset
    fn sample_coreset(&mut self) {
    } // end of sample_coreset
} // end of impl block
