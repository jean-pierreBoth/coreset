//! Implementation coreset sensitivity sampling.
//!
//!   - New Fraweworks for Offline and Streaming Coreset Constructions.  
//!        Braverman, Feldman, Lang, Statsman 2022
//!        [arxiv-v3](https://arxiv.org/abs/1612.00889)

// We need 2 passes on data as Bmor algorithm can merge data when rescaling cost and facility number so data id are not conserved.

use anyhow::*;

use rayon::prelude::*;
use std::sync::Arc;

use dashmap::DashMap;
use std::collections::hash_map; // for key() method
use std::collections::HashMap;
use std::hash::Hash;

use ndarray::{Array1, Array2};

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use crate::bmor::*;
use crate::discrete::DiscreteProba;
use crate::facility::*;
use crate::makeiter::*;

use anndists::dist::*;

#[derive(Copy, Clone, Debug)]
// a sampled point
struct Point<DataId> {
    pub(self) id: DataId,
    pub(self) _rank: usize,
    pub(self) proba: f64,
}

struct PointSampler<DataId> {
    /// proba distribution over points
    proba: DiscreteProba<f64>,
    // first usize (key) is an index in self.proba, second DataId (value) is data_id in data (possibly an index).
    w_index: HashMap<usize, DataId>,
}

impl<DataId: Clone> PointSampler<DataId> {
    fn new(weights: &Vec<f64>, w_index: HashMap<usize, DataId>) -> Self {
        PointSampler {
            proba: DiscreteProba::new(&weights),
            w_index,
        }
    }

    /// sample a random data point, returning its Id (possibly an index in some array)
    fn sample<R>(&self, rng: &mut R) -> Point<DataId>
    where
        R: Rng,
    {
        let (rank, proba) = self.proba.sample(rng);
        let opt_id = self.w_index.get(&rank);
        if opt_id.is_none() {
            log::error!("no id for rank sampled : {}", rank);
            std::panic!();
        }
        let id = opt_id.cloned().unwrap();
        return Point {
            id: id,
            _rank: rank,
            proba,
        };
    } // end of sample
} // end of PointSampler

//======================================================================================================

// How do we represent a coreset: For now minimal
/// Structure representing Coreset obtained with coreset construction algorithms
/// It stores for each coreset point its id and a Vector of associated weights with which the points appears in coreset.
pub struct CoreSet<DataId, T: Send + Sync + Clone, Dist: Distance<T> + Clone + Sync + Send> {
    // maps id to weight/multiplicity
    id_weight_map: HashMap<DataId, f64>,
    // stores couples (id, data vector). Stored in the order they retrieved not same order as id_w_map
    datas_wid: Option<Vec<(DataId, Vec<T>)>>,
    //
    distance: Dist,
} // end of Coreset

impl<DataId, T: Send + Sync + Clone, Dist> CoreSet<DataId, T, Dist>
where
    DataId: Eq + Hash + Send + Sync + Clone,
    Dist: Distance<T> + Clone + Sync + Send,
{
    pub fn new(
        core_w: HashMap<DataId, f64>,
        datas_wid: Option<Vec<(DataId, Vec<T>)>>,
        distance: Dist,
    ) -> CoreSet<DataId, T, Dist> {
        assert_eq!(core_w.len(), datas_wid.as_ref().unwrap().len());
        CoreSet {
            id_weight_map: core_w,
            datas_wid,
            distance,
        }
    }

    /// returns number of different points
    pub fn get_nb_points(&self) -> usize {
        self.id_weight_map.len()
    }

    /// returns weight of a given point if present in coreset
    pub fn get_weight(&self, data_id: DataId) -> Option<f64> {
        let index_res = self.id_weight_map.get(&data_id);
        match index_res {
            Some(index) => {
                return Some(*index);
            }
            _ => {
                return None;
            }
        }
    } // end of get_weight

    /// returns an iterator on the id of data
    pub fn get_data_ids(&self) -> hash_map::Keys<DataId, f64> {
        return self.id_weight_map.keys();
    }

    /// get an iterator on couples (id, weight)
    pub fn get_items(&self) -> hash_map::Iter<DataId, f64> {
        self.id_weight_map.iter()
    }

    /// for a coreset point of rank r returns id and data vector.
    pub(crate) fn get_point_by_rank(&self, r: usize) -> Option<(DataId, &Vec<T>)> {
        let res = match self.datas_wid.as_ref() {
            Some(v) => {
                if r < v.len() {
                    Some((v[r].0.clone(), &v[r].1))
                } else {
                    log::error!(
                        "get_point_by_rank could not find data vector r : {} size : {}",
                        r,
                        v.len()
                    );
                    std::panic!("get_point_by_rank could not find data vector r");
                }
            }
            None => {
                log::error!("get_point_by_rank could not find data vector r : {}", r);
                std::panic!("get_point_by_rank could not find data vector r");
            }
        };
        //
        return res;
    } // end of get_point

    /// computes matrix distances between points.
    /// line i of matrix corresponds to id in the Vec\<usize\> i'th element of first argument of the option returned
    ///
    pub fn compute_distances(&self) -> Option<(Vec<DataId>, Array2<f32>)> {
        let nbpoints = self.get_nb_points();
        // allocates to zero rows. We will computes rows in //
        let mut distances = Array2::<f32>::zeros((0, nbpoints));
        //
        let compute_row = |i| -> Array1<f32> {
            let mut row_i = Array1::zeros(nbpoints);
            let (_, p_i) = self.get_point_by_rank(i).unwrap();
            for j in 0..nbpoints {
                let p_j = self.get_point_by_rank(j).unwrap().1;
                if j != i {
                    row_i[j] = self.distance.eval(p_i, p_j);
                }
            }
            return row_i;
        };
        //
        //
        let rows: Vec<(usize, Array1<f32>)> = (0..nbpoints)
            .into_par_iter()
            .map(|i| (i, compute_row(i)))
            .collect();
        // now we have rows we must transfer into distances
        for (r, v) in &rows {
            assert_eq!(*r, distances.shape()[0]);
            distances.push_row(v.into()).unwrap();
        }
        //
        let ids: Vec<DataId> = self
            .datas_wid
            .as_ref()
            .unwrap()
            .iter()
            .map(|(id, _)| (*id).clone())
            .collect();
        //
        return Some((ids, distances));
    } // end of compute_distances
} // end of impl Coreset

/// This structure provides Algorithm1 Braverman and al 2022, implementing Sensitivity sampling.  
/// It relies on [bmor](super::bmor) algorithm.  
/// The algorithm needs  one streaming pass and one sampling pass.  
/// The data must be given consistent id across the 2 passes. (The data id can be its rank in the stream in which case the 2 pass
/// must process data in the same order)
pub struct Coreset1<DataId, T: Send + Sync + Clone, Dist: Distance<T> + Clone + Sync + Send> {
    ///
    phase: usize,
    /// keep track of number of data processed
    nb_data: usize,
    /// bmor instance
    bmor: Bmor<DataId, T, Dist>,
    /// facilities with respect to which we compute sensitivity (or importance)
    facilities: Option<Facilities<DataId, T, Dist>>,
    // A map to store facility associated to each point. Needed in sensitivity
    point_facility_map: Option<Arc<DashMap<DataId, PointMap>>>,
} // end of Coreset1

// s estimation

impl<DataId, T: Send + Sync + Clone, Dist> Coreset1<DataId, T, Dist>
where
    Dist: Distance<T> + Clone + Sync + Send,
    DataId: Eq + Hash + std::fmt::Debug + Clone + Send + Sync,
{
    /// nbdata_expected : the (expected) data size
    /// k  : The expected number of facilities
    /// distance : the metric to use.
    /// beta and gamma are arguments of Bmor
    pub fn new(k: usize, nbdata_expected: usize, beta: f64, gamma: f64, distance: Dist) -> Self {
        let bmor = Bmor::new(k, nbdata_expected, beta, gamma, distance);
        let phase = 0usize;
        //
        Coreset1 {
            phase,
            nb_data: 0,
            bmor,
            facilities: None,
            point_facility_map: None,
        }
    } // end of new

    /// The main interface to the algorithm.  
    ///
    /// The size of the coreset generated will be about fraction * size of data.
    /// In fact a point can be sampled many times, but in this case the sampled points are merged and their weight added.
    /// So the fraction should be set to a value slightly superior to the one desired.  
    /// A value of 0.11 is a good initial guess to get a fraction of 0.1
    pub fn make_coreset<IterGenerator>(
        &mut self,
        iter_generator: &IterGenerator,
        fraction: f64,
    ) -> anyhow::Result<CoreSet<DataId, T, Dist>>
    where
        IterGenerator: MakeIter<Item = (DataId, Vec<T>)>,
        DataId: Eq + Hash + std::fmt::Debug + Send + Sync,
    {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
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
        let id_weight_map = self.sample_coreset(&sampler, fraction);
        let distance = self.facilities.as_ref().unwrap().get_distance();
        // now we have ids and weights of points in coreset but we need a last pass to store the data associated to id!
        let id_data_map = self.retrieve_corepoints_by_id(&id_weight_map, iter_generator);
        //
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            "\n Coreset1::make_coreset  sys time(ms) {:?} cpu time(ms) {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        //
        Ok(CoreSet::new(
            id_weight_map,
            Some(id_data_map),
            distance.clone(),
        ))
    } // end of make_coreset

    // we need to retrieve the data vector corresponding to the id of coreset points
    // Careful , the data are stored in the order they are found by iter_generator and not in the order of the HashMap
    fn retrieve_corepoints_by_id<IterGenerator>(
        &self,
        id_weight_map: &HashMap<DataId, f64>,
        iter_generator: &IterGenerator,
    ) -> Vec<(DataId, Vec<T>)>
    where
        IterGenerator: MakeIter<Item = (DataId, Vec<T>)>,
        DataId: Eq + Hash + std::fmt::Debug,
    {
        //
        let mut iter = iter_generator.makeiter();
        //
        let nbpoints = id_weight_map.len();
        let mut datas_wid: Vec<(DataId, Vec<T>)> = Vec::with_capacity(nbpoints);
        while let Some((id, data)) = iter.next() {
            // do we need the data of this id
            if id_weight_map.contains_key(&id) {
                datas_wid.push((id, data));
            }
        }
        // temporary what is the id not found (bug fixed !)
        if datas_wid.len() < nbpoints {
            let mut set = indexmap::IndexSet::with_capacity(nbpoints + 100);
            for (id, _) in &datas_wid {
                set.insert(id);
            }
            for (id, _) in id_weight_map {
                // do we have id in set
                if set.get(id).is_none() {
                    log::error!(" we do not have id : {:?} in set", id);
                }
            }
        }
        //
        assert_eq!(datas_wid.len(), nbpoints);
        //
        return datas_wid;
    } // end of retrieve_corepoints_by_id

    /// This function takes an iterator on all data and process (with buffering and parallelizing) them via calling *process_data()* , consuming the iterator
    fn process_data_iterator(
        &mut self,
        mut iter: impl Iterator<Item = (DataId, Vec<T>)>,
    ) -> anyhow::Result<()> {
        // TODO: adapt bufsize to memory/cpu
        let bufsize: usize = 50000;
        let mut datas = Vec::<Vec<T>>::with_capacity(bufsize);
        let mut ids = Vec::<DataId>::with_capacity(bufsize);
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
    fn process_data(&mut self, data: &[Vec<T>], data_id: &[DataId]) -> anyhow::Result<()> {
        //
        if self.phase == 0 {
            let _ = self.bmor.process_data(&data, data_id).unwrap();
            self.bmor.log();
            self.nb_data += data.len();
            return Ok(());
        } else {
            let facilities_ref = self.facilities.as_ref().unwrap();
            //
            let dispatch_i = |item: usize| {
                // get facility rank and weight
                let (facility, dist) = facilities_ref
                    .get_nearest_facility(&data[item], false)
                    .unwrap();
                let weight = 1.;
                self.facilities
                    .as_ref()
                    .unwrap()
                    .insert_point(facility, dist, weight);
                match &self.point_facility_map {
                    Some(f_map) => {
                        let p_map = PointMap::new(facility, dist, 1.);
                        let res = f_map.insert(data_id[item].clone(), p_map);
                        if res.is_some() {
                            log::error!("data_id {:?} is already present error", data_id[item]);
                            std::panic!();
                        }
                        log::trace!(
                            "inserted PointMap for data_id {:?} in facility map",
                            data_id[item]
                        );
                    }
                    _ => {
                        std::panic!("no facility_map allocated, should not happen")
                    }
                }
            };
            // now we insert into pointmap if necessary
            (0..data.len())
                .into_par_iter()
                .for_each(|item| dispatch_i(item));
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
                let _ = self.facilities.as_mut().unwrap().compute_weight_cost();
                self.facilities.as_ref().unwrap().log(0);
            }

            _ => {
                std::panic!("should not occurr");
            }
        }
    } //

    /// returns a reference to (optional) facility_map
    pub fn get_facility_map(&self) -> Option<&Arc<DashMap<DataId, PointMap>>> {
        return self.point_facility_map.as_ref();
    }

    /// returns the number of data (know after end of first pass)
    pub fn get_nb_data(&self) -> usize {
        self.nb_data
    }

    // if needed (as in the case of sensitivity computations) allocate
    fn init_facility_map(&mut self, capacity: usize) {
        self.point_facility_map = Some(Arc::new(DashMap::with_capacity(capacity)))
    }

    //  initialize point_weights : Option<WeightedIndex<usize>>,
    // The function receive an iterator over data.
    // from rust 1.75 such an iterator can be obtained with the user implementing a trait providing the iterator on data
    fn build_sampling_distribution(&mut self) -> PointSampler<DataId> {
        // The 2 denonimators used in line 3 of algo 1 for Coreset in Braverman
        let facilities_ref = self.facilities.as_ref().unwrap();
        // denominator used in line 3  of algo 1 for Coreset in Braverman
        let global_cost = facilities_ref.get_cost();
        log::info!(
            "build_sampling_distribution got global cost : {:.3e}",
            global_cost
        );
        let p_facility_map_ref = self.point_facility_map.as_ref().unwrap();
        let nb_facilities = facilities_ref.len(); // This is |B| in line 3  of algo 1 for Coreset in Braverman
        let mut cumul_proba = 0.;
        // the fields to build PointSampler
        let mut p_weights = Vec::<f64>::with_capacity(self.nb_data);
        let mut w_index = HashMap::<usize, DataId>::with_capacity(self.nb_data);
        // we iter on reviously built p_facility_map_ref
        let mut pmap_iter = p_facility_map_ref.iter();
        while let Some(iter_ref) = pmap_iter.next() {
            let (dataid, pointmap) = iter_ref.pair();
            //
            let mut proba = pointmap.get_dist() as f64 * pointmap.get_weight() as f64 / global_cost;
            // get weight of facility of point corresponding to data_id
            let f_weight = facilities_ref.get_facility_weight(pointmap.get_facility());
            proba =
                proba + pointmap.get_weight() as f64 / (nb_facilities as f64 * f_weight.unwrap());
            proba = proba * 0.5;
            // now we can update
            assert!(proba > 0.);
            p_weights.push(proba);
            // key is rank, value is id!!!
            w_index.insert(p_weights.len() - 1, dataid.clone());
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
    fn sample_coreset(
        &mut self,
        sampler: &PointSampler<DataId>,
        rate: f64,
    ) -> HashMap<DataId, f64> {
        //
        log::info!("sample_coreset fraction : {:.2e}", rate);
        let nb_sample = (rate * self.nb_data as f64) as usize;
        //
        let mut coreset = HashMap::<DataId, f64>::with_capacity(2 * nb_sample);
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(14537);
        //
        for _ in 0..nb_sample {
            let point = sampler.sample(&mut rng);
            let weight = 1. / (point.proba * nb_sample as f64);
            let id_weights = coreset.get_mut(&point.id);
            match id_weights {
                Some(old_weight) => {
                    *old_weight += weight;
                }
                None => {
                    coreset.insert(point.id, weight);
                }
            }
        } // end of for
          //
        log::info!(
            "sensitivity::sample_coreset coreset nb points :  {}",
            coreset.len()
        );
        coreset
    } // end of sample_coreset
} // end of impl block
