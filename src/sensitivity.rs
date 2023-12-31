//! Implementation sensitivity sampling as described in:
//! 
//!   - New Fraweworks for Offline and Streaming Coreset Constructions
//!        Braverman, Feldman, Lang, Statsman 2022
//!        [arxiv-v3](https://arxiv.org/abs/1612.00889)

// We need 2 passes on data as Bmor algorithm can merge data when rescaling cost and facility number so data id are not conserved.

//#![allow(unused)]

use anyhow::*;

use std::sync::Arc;
use rayon::prelude::*;

use std::collections::HashMap;
use std::collections::hash_map;
use dashmap::DashMap;

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;


use crate::iterprovider::*;
use crate::bmor::*;
use crate::facility::*;
use crate::discrete::DiscreteProba;

use hnsw_rs::dist::*;

// a sampled point 
struct Point {
    pub(self) id : usize,
    pub(self) _rank : usize,
    pub(self) proba : f32,
}

struct PointSampler {
    /// proba distribution over points
    proba : DiscreteProba<f32>,
    // first usize is an index in point_weights, second index is data_id in data (possibly an index).
    w_index : HashMap<usize, usize>,
}

impl PointSampler {

    fn new(weights : &Vec<f32>, w_index : HashMap<usize, usize>) -> Self {
        PointSampler{proba : DiscreteProba::new(&weights) , w_index}
    }

    /// sample a random data point, returning its Id (possibly an index in some array)
    fn sample<R>(&self, rng : &mut R ) -> Point
        where R : Rng  {
            let (rank, proba) = self.proba.sample(rng);
            let id = *self.w_index.get(&rank).unwrap();
            return Point{id, _rank : rank, proba}
    } // end of sample


} // end of PointSampler


//======================================================================================================

// How do we represent a coreset: For now minimal
/// Structure representing Coreset obtained with coreset construction algorithms
pub struct CoreSet {
    core : HashMap<usize, Vec<f32>>,
} // end of Coreset



impl CoreSet {

    pub fn new(core : HashMap<usize, Vec<f32>>) -> Self {
        CoreSet{core}
    }

    /// returns number of different points
    pub fn get_nb_points(&self) -> usize {
        self.core.len()
    }

    /// returns list of weights of a given point if present in coreset
    pub fn get_weights(&self, data_id : usize) -> Option<&Vec<f32>> {
        let index_res = self.core.get(&data_id);
        match index_res {
            Some(index) => {
                return Some(index);
            }
            _ => { return None; }
        }
    } // end of get_weights

    /// returns the id of data
    pub fn get_data_ids(&self) -> hash_map::Keys<usize, Vec<f32>> { return self.core.keys()}

    /// total number of points, taking into account multiplicity
    pub fn get_size(&self) -> usize {
        let size = self.core.iter().map(|(_,v)| v.len()).sum();
        return size;
    }
} // end of impl Coreset




/// This structure provides Algorithm1 Braverman and al 2022, it relies on [bmor](super::bmor) algorithm.  
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
    // A map to store facility associated to each point. Needed in sensitivity
    point_facility_map : Option<Arc<DashMap<usize,PointMap>>>,
} // end of Coreset1



// s estimation 

impl <T:Send+Sync+Clone, Dist> Coreset1<T, Dist> 
            where Dist : Distance<T> + Clone + Sync + Send {
    /// k, beta and gamma are arguments of Bmor 
    pub fn new(k: usize, nbdata_expected : usize, beta : f64, gamma : f64, distance :  Dist) -> Self {
        let bmor = Bmor::new(k, nbdata_expected, beta, gamma, distance);
        let phase = 0usize;
        //
        Coreset1{ phase, nb_data : 0, bmor, facilities : None, point_facility_map : None}
    } // end of new


    /// main interface to the algorithm
    pub fn make_coreset<IterGenerator>(&mut self, iter_generator : &IterGenerator) ->  anyhow::Result<CoreSet> 
        where IterGenerator : IterProvider<DataType = (usize, Vec<T>)> {
        // first bmor pass to get a list of facilities
        let iter = iter_generator.makeiter();
        let res1 = self.process_data_iterator(iter);
        if res1.is_err() {
            log::error!("first pass failed");
            return Err(anyhow!("first pass failed"));
        }
        // In phase 2, we have facilities, we empty them and redispatch data and store the facility of each point
        log::info!("end of first pass, second pass to compute point facility map");
        self.facilities.as_mut().unwrap().empty();
        self.init_facility_map(self.nb_data);
        let iter = iter_generator.makeiter();
        let res2 = self.process_data_iterator(iter);
        if res2.is_err() {
            log::error!("seond pass failed");
            return Err(anyhow!("second pass failed"));
        }
        // now we have info for building sampling distribution in self.p_facility_map
        log::info!("end of second pass, doing sensitivity and sampling computations");
        let sampler = self.build_sampling_distribution();
        // we can now get rid of p_facility_map
        self.point_facility_map = None;
        //
        let coreset = self.sample_coreset(&sampler);
        Ok(CoreSet::new(coreset))
    }  // end of make_coreset


    /// This function takes an iterator on all data and process (with buffering and parallelizing) them via calling *process_data()* , consuming the iterator
    pub fn process_data_iterator(&mut self, mut iter : impl Iterator<Item=(usize, Vec<T>)>) -> anyhow::Result<()> {
        // TODO: adapt bufsize to memory/cpu
        let bufsize : usize = 50000;
        let mut datas = Vec::<Vec<T>>::with_capacity(bufsize);
        let mut ids = Vec::<usize>::with_capacity(bufsize);
        //
        loop {
            let data_opt = iter.next();
            match data_opt {
                Some((id, data)) => {
                    // insert
                    datas.push(data);
                    ids.push(id);
                    if datas.len() == bufsize {
                        // process
                        let res = self.process_data(&datas, &ids);
                        assert!(res.is_ok());
                        // empty buffer
                        datas.clear();
                        ids.clear();
                    }
                }
                _ => {
                    if datas.len() > 0 {
                        let res = self.process_data(&datas, &ids);
                        assert!(res.is_ok());
                        // empty buffer
                        datas.clear();
                        ids.clear();  
                    }
                    break;
                }
            } // end match
        } // end loop
        // DO NOT FORGET calling end_pass
        self.end_pass();
        //
        return Ok(());
    } // end of process_data_iterator


    /// treat unweighted data. 
    /// This functions provides a buffered, parallelized internal implementation of process_data_iterator.   
    /// At end of first round on data [end_pass](Self::end_pass()) must be called before running the second pass on data
    fn process_data(&mut self, data : &[Vec<T>], data_id : &[usize]) -> anyhow::Result<()> {
        //
        if self.phase == 0 {
            let _ = self.bmor.process_data(&data, data_id).unwrap();
            self.bmor.log();
            self.nb_data += data.len();
            return Ok(());
        }
        else {
            let facilities_ref = self.facilities.as_ref().unwrap();
            //
            let dispatch_i = | item : usize | {
                // get facility rank and weight
                let (facility, dist) = facilities_ref.get_nearest_facility(&data[item], false).unwrap();           
                let weight = 1.;
                self.facilities.as_ref().unwrap().insert_point(facility, dist, weight);
                match &self.point_facility_map {
                    Some(f_map) => {
                        let p_map = PointMap::new(facility, dist, 1.);
                        let res = f_map.insert(data_id[item], p_map);
                        if res.is_some() {
                            log::error!("data_id {} is already present error", data_id[item]);
                            std::panic!();
                        }
                        log::trace!("inserted PointMap for data_id {} in facility map", data_id[item]);
                    }
                    _  => { std::panic!("no facility_map allocated, should not happen") }
                }          
            };
            // now we insert into pointmap if necessary
            (0..data.len()).into_par_iter().for_each( |item| dispatch_i(item));
            //
            return Ok(());
        };
     } // end of process_data





    /// declares end of streaming data first pass, and construct coreset by sampling
    fn end_pass(&mut self) {
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
                self.facilities.as_ref().unwrap().log(0);
            }

            _ => {
                std::panic!("should not occurr");
            }
        }
    } // 


    /// returns a reference to (optional) facility_map
    pub fn get_facility_map(&self) -> Option<&Arc<DashMap<usize, PointMap>>> {
        return self.point_facility_map.as_ref()
    }


    // if needed (as in the case of sensitivity computations) allocate 
    fn init_facility_map(&mut self, capacity : usize) {
            self.point_facility_map = Some(Arc::new(DashMap::with_capacity(capacity)))
    }

    // if there is a p_facility_map we store pointmap
    fn insert_pointmap(&self, id : usize, pointmap : PointMap) -> anyhow::Result<()> {
        let res = match &self.point_facility_map {
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
    fn build_sampling_distribution(&mut self) -> PointSampler {
        // The 2 denonimators used in line 3 of algo 1 for Coreset in Braverman
        let facilities_ref = self.facilities.as_ref().unwrap();
        // denominator used in line 3  of algo 1 for Coreset in Braverman
        let global_cost = facilities_ref.get_cost();
        log::info!("build_sampling_distribution got global cost : {:.3e}", global_cost);
        let p_facility_map_ref = self.point_facility_map.as_ref().unwrap();
        let nb_facilities = facilities_ref.len();     // This is |B| in line 3  of algo 1 for Coreset in Braverman
        let mut cumul_proba = 0.;
        // the fields to build PointSampler
        let mut p_weights = Vec::<f32>::with_capacity(self.nb_data);
        let mut w_index = HashMap::<usize, usize>::with_capacity(self.nb_data);
        // we iter on reviously built p_facility_map_ref
        let mut pmap_iter = p_facility_map_ref.iter();
        while let Some(iter_ref) = pmap_iter.next() {
            let (dataid, pointmap) = iter_ref.pair();
            let mut proba = pointmap.get_dist() as f64 * pointmap.get_weight() as f64 / global_cost;
            // get weight of facility of point corresponding to data_id
            let f_weight = facilities_ref.get_facility_weight(pointmap.get_facility());
            proba = proba + pointmap.get_weight() as f64 / (nb_facilities as f64 *  f_weight.unwrap());
            proba = proba * 0.5;
            // now we can update 
            assert!(proba > 0.);
            p_weights.push(proba as f32);
            w_index.insert(*dataid, p_weights.len());
            // for a check
            cumul_proba += proba;
        }
        log::debug!("cumul_proba : {:.5e}", cumul_proba);
        assert!((1. - cumul_proba).abs() < 1.0E-5);
        let point_sampler = PointSampler::new(&p_weights, w_index);
        //
        return point_sampler;
    } // end of build_sampling_distribution



    // build and init field coreset
    fn sample_coreset(&mut self, sampler : &PointSampler) -> HashMap::<usize, Vec<f32>> {
        // TODO: determine how many point we need to sample
        let nb_sample = 1000;
        //
        let mut coreset = HashMap::<usize, Vec<f32>>::with_capacity(nb_sample);
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(14537);
        //
        for i in 0..nb_sample {
            let point = sampler.sample(&mut rng);
            let weight =  1./ (point.proba * (i as f32));
            let id_weights = coreset.get_mut(&point.id);
            match id_weights {
                Some(weights) => {
                    weights.push(weight);
                }
                None => {
                    let mut weights = Vec::<f32>::new();
                    weights.push(weight);
                    coreset.insert(point.id, weights);
                }
            }
        } // end of for 
        coreset
    } // end of sample_coreset


} // end of impl block
