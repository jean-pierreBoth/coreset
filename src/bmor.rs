//! Adaptation of Streaming k-means on well clustered data.  
//! Braverman Meyerson Ostrovski Roytman ACM-SIAM 2011 [braverman-2](https://dl.acm.org/doi/10.5555/2133036.2133039)
//! 
//! **This algorithm can run in a streaming context**.  
//! 
//! We do not constrain the clustering output to be exactly some value k but let the number of clusters be
//! the result of the main algorithms.   
//! The final of number of facilities can be reduced by running an end step.  
//! **Bmor algorithm dispatch points on the fly so it computes an upper bound of the cost**.  
//! **But it is possible to [dispatch_data](crate::facility::Facilities::dispatch_data()) explicitly** 
//! 
//! This algorithm can process mnist fashion data in 1 second on a i9 laptop (without requiring heavy multithreading)
//! 
//! 


use std::marker::PhantomData;

use parking_lot::RwLock;
use std::sync::Arc;
use std::cell::RefCell;

use anyhow::anyhow;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;
use rand::distributions::{Distribution,Uniform};

use hnsw_rs::dist::*;

use crate::facility::*;


/// This structure stores the state of Bmor algorithm through iterations.
/// In particular it stores allocated facilities.
#[derive(Clone)]
pub struct BmorState<DataId, T:Send+Sync+Clone, Dist : Distance<T> > {
    // (1+logn)k
    oneplogn : usize,
    // nb iterations (phases)
    phase : usize,
    // initial cost factor
    li : f64,
    // at each phase we have an upper bound for cost.
    phase_cost_upper : f64,
    // upper bound on number of facilities
    facility_bound : usize, 
    // current centers, associated to rank in stream (or in data) and weight (nb points in facility)
    centers : Facilities<DataId, T, Dist>,
    // sum of absolute value (some algos have <0 weights) of inserted weight
    absolute_weight : f64,
    // total cost
    total_cost : f64,
    //
    nb_inserted : usize,
    //
    rng : Xoshiro256PlusPlus,
    //
    unif : Uniform::<f64>,
} // end of 


impl<DataId : std::fmt::Debug + Clone + Send + Sync, T:Send+Sync+Clone, Dist : Distance<T> + Clone + Sync + Send> BmorState<DataId, T, Dist> {

    pub(crate) fn new(k : usize, nbdata : usize, phase : usize, alloc_size : usize, upper_cost : f64, facility_bound : usize, distance : Dist) -> Self {
        let centers = Facilities::<DataId, T, Dist>::new(alloc_size, distance);
        let unif = Uniform::<f64>::new(0., 1.);
        let rng = Xoshiro256PlusPlus::seed_from_u64(1454691);
        let oneplogn = (1 + nbdata.ilog2()) as usize * k;
        let li = 1.0f64;
        //
        log::info!("BmorState creation : facility bound : {:?}", facility_bound);
        //
        BmorState{oneplogn, phase, li, phase_cost_upper : upper_cost, facility_bound, centers, absolute_weight : 0., total_cost : 0., nb_inserted : 0, rng, unif}
    }

    /// returns facilities as computed by the algorithm
    pub fn get_facilities(&self) -> &Facilities<DataId, T, Dist> {
        return &self.centers
    }

    /// returns a mutable reference to facilities (useful for calling [dispatch_labesl](crate::facility::Facilities::dispatch_data())).
    pub fn get_mut_facilities(&mut self) -> &mut Facilities<DataId, T, Dist> {
        return &mut self.centers
    }

    // get current phase num of processing
    pub fn get_phase(&self) -> usize {
        self.phase
    }

    #[allow(unused)]
    pub(crate) fn get_li(&self) -> f64 {
        self.li
    }


    pub(crate) fn get_nb_inserted(&self) -> usize {
        self.nb_inserted
    }


    pub(crate) fn get_unif_sample(&mut self) -> f64 {
        self.unif.sample(&mut self.rng)
    }

    pub(crate) fn get_phase_cost_bound(&self) -> f64 {
        self.phase_cost_upper
    }

    /// get upper bound  for number of facilities
    #[allow(unused)]
    pub(crate) fn get_facility_upper_bound(&self) -> usize {
        self.facility_bound
    }

    /// get sum of absolute value of weights inserted
    pub(crate) fn get_weight(&self) -> f64 {
        self.absolute_weight
    }

    /// get sum of absolute value of weights inserted
    pub(crate) fn get_cost(&self) -> f64 {
        self.total_cost
    }    

    /// get nearest center/facility of a point, its rank and distance to facility
    pub fn get_nearest_center(&self, point : &[T]) -> Option<(&Arc<RwLock<Facility<DataId, T>>>, usize, f32) >
        where T : Send + Sync,  Dist : Sync {
        //
        let nb_facility = self.centers.len();
        //
        if nb_facility == 0 {
            return None;
        }
        // get nearest facilty
        let (rank,dist) = self.centers.get_nearest_facility(point, false).unwrap();
        //
        return Some( (self.centers.get_facility(rank).unwrap(), rank, dist));
    } // end of get_nearest_center


    /// insert into an already existing facility
    /// return true if all is OK, false if costs or number of facilities got too large
    fn update(&mut self, rank_id : DataId, point : &[T], weight : f64) -> bool {
        //
        log::trace!("in BmorState::update rank_id: {:?}", rank_id);
        //
        let dist_to_nearest : f32;
        let nearest_facility : Arc<RwLock<Facility<DataId, T>>>;
        {
            let nearest_facility_res = self.get_nearest_center(point);
            if nearest_facility_res.is_none() {
                log::error!("internal error, update did not find nearest facility");
                std::process::exit(1);
            }
            let nearest_center =  nearest_facility_res.unwrap();
            dist_to_nearest = nearest_center.2;
            nearest_facility = nearest_center.0.clone();
        }
        // take into account f factor
        if self.get_unif_sample() < (weight * dist_to_nearest as f64 * self.oneplogn as f64 / self.li) {
            // we create a new facility. No cost increment
            let mut new_f = Facility::<DataId, T>::new(rank_id, point);
            new_f.insert(weight as f64,0.);
            self.centers.insert(new_f);
            // log::debug!("in BmorState::update  creating new facility around {}, nb_facilities : {}", rank_id, self.centers.len());
        }
        else {
            // log::debug!("in BmorState::update rank_id: {:?}, inserting in old facility dist : {:.3e}", rank_id, dist_to_nearest);
            nearest_facility.write().insert(weight, dist_to_nearest);
            self.total_cost += weight.abs() as f64 * dist_to_nearest as f64;
        }
        // we increments weight monitoring and number of insertions
        self.absolute_weight += weight.abs() as f64;
        self.nb_inserted += 1;
        // check if we are above constraints
        if self.total_cost > self.phase_cost_upper || self.centers.len() > self.facility_bound {
            if log::log_enabled!(log::Level::Debug) {
                log::debug!("constraint violation");
                self.log();
            }
            return false
        }
        else {
            return true;
        }
    } // end of update


    // reinitialization. (upper cost rescaling)
    pub(crate) fn reinit(&mut self, beta : f64) {
        self.phase += 1;
        self.phase_cost_upper *= beta;
        self.li *= beta;
        self.centers.clear();
        self.absolute_weight = 0.;
        self.total_cost = 0.;
    }

    pub(crate) fn log(&self) {
        log::debug!("\n\n BmorState::log_state");
        log::debug!("\n nb facilities : {:?}", self.centers.len());
        log::debug!("\n weight : {:.3e}   cost {:.3e}", self.get_weight(), self.get_cost());
        log::debug!("\n nb facility max : {:?}, upper cost bound : {:.3e}", self.facility_bound, self.get_phase_cost_bound());
        log::info!("\n nb total insertion : {:?}  nb_phases: {:?}", self.get_nb_inserted(), self.phase + 1);
    }
} // end of impl block BmorState



#[cfg_attr(doc, katexit::katexit)]
/// This structure gathers all parameters defining Bmor algorithm.  
/// The algorithm do iterations with at each step an acceptable upper bound cost and upper bound on number
/// facilities. The upper bounds are increased if iteration constraints are not satisfied, exisiting facilities are recycled as
/// old points and the algortitm can go on with new incoming points in a streaming way.
/// 
/// These upper bounds are defined using 2 parameters : $ \beta $ and $ \gamma $.  
/// 
/// let $k$ be the number of expected facilities (centers),  the upper bound on number facilities is
/// defined by : $ (\gamma −1) \space k \space (1+ \log_2 n)$.  
/// At each iteration $i$ the upper bound of cost $C_{i}$ is defined  by $ \beta * C_{i-1} $ and the allocation of a facility 
/// is relaxed in a coherent way.  
/// As for large n the resulting number of allocated facilities can be larger than k it is possible to ask for an end step [end_step](Self::end_data()) that 
/// will reduce the number of facilities to less than $ (\gamma −1) \space k \space (1+ \log_2 nbfacility)$
/// 
/// The data are affected to a facility on the fly (useful in streaming). 
/// But it is possible for a point to be nearer to a facility opened later with data arriving after it.  
/// So the dispatching cost can be optimized a posteriori (in a second pass on the data) with method [dispatch_data](crate::facility::Facilities::dispatch_data())
///   
/// 
/// $\beta$ and $\gamma$ can be initialized by 2.
pub struct Bmor<DataId, T:Send+Sync+Clone, Dist : Distance<T> > {
    // base number of centers expected
    k : usize,
    //
    nbdata_expected : usize,
    // cost multiplicative factor for upper bound of accepted cost at each phase.
    beta : f64,
    //  slackness parameters for cost and number of centers accepted
    gamma : f64,
    //
    distance : Dist,
    // store computation state 
    state : RefCell<BmorState<DataId, T,Dist>>,
    //
    _t : PhantomData<T>,
}  // end of struct Bmor



impl <DataId, T : Send + Sync + Clone, Dist> Bmor<DataId, T, Dist> 
    where     Dist : Distance<T> + Clone + Sync + Send,
            DataId : std::fmt::Debug + Clone + Send + Sync {

    /// - k: number of centers.  
    /// - nbdata : nb data expected. As this algorithm can be used in streaming (successive calls to methods [process_data](Self::process_data()) or
    ///     [process_weighted_data](Self::process_weighted_data())) the number of expected data can be larger than the length or arguments passed to these methods.
    /// - beta : upper cost multiplicative factor
    /// - gamma : slackness factor for number facilities upper bound.
    /// - end_step : if true a second step is done to further reduc the number of facilities.
    ///         
    pub fn new(k: usize, nbdata_expected : usize, beta : f64, gamma : f64, distance :  Dist) -> Self {
        // TODO: to be adapted?
        let nb_centers_bound = ((gamma - 1.) * (1. + nbdata_expected.ilog2() as f64) * k as f64).trunc() as usize; 
        let upper_cost = gamma;
        let state = BmorState::<DataId, T, Dist>::new(k, nbdata_expected, 0, nb_centers_bound as usize, 
            upper_cost as f64, nb_centers_bound, distance.clone());
        //
        Bmor{k, nbdata_expected, beta, gamma, distance, state: RefCell::new(state),  _t : PhantomData::<T> }
    }

    /// return expected number of facilities (clusters)
    pub fn get_k(&self) -> usize { self.k}

    /// get_beta
    pub fn get_beta(&self) -> f64 { self.beta} 

    /// get gamma
    pub fn get_gamma(&self) -> f64 { self.gamma}

    /// treat unweighted data. 
    /// **This method can be called many times in case of data streaming, passing data by blocks**.  
    /// It returns the number of facilities created up to this call.
    /// id are data id (anything identifying data point)
    pub fn process_data(&mut self, data : &[Vec<T>], id : &[DataId]) -> anyhow::Result<usize> {
        //
        let weighted_data: Vec<(f64, &Vec<T>, DataId)> = (0..data.len()).into_iter().map( |i| (1.,&data[i],id[i].clone())).collect();
        self.process_weighted_block(&weighted_data);
        //
        let state = self.state.borrow();
        state.log();
        if log::log_enabled!(log::Level::Debug) {
            state.get_facilities().log(1);
        }
        //
        return Ok(state.get_facilities().len());
    } // end of process_data

    // 
    /// declare end of streaming data.
    /// This method returns the facilities created.
    /// if contraction flag is set to true, a final pass of the bmor algorithm will be used to try to reduce the
    /// number of facilities created by previous call to [process_data](Self::process_data()) or [process_weighted_data](Self::process_weighted_data())
    pub fn end_data(&self, contraction : bool) -> Facilities<DataId, T, Dist> {
        let facilities = match contraction {
            false => {
                let facilities_ret = self.state.borrow().get_facilities().clone();
                facilities_ret.log(0);
                facilities_ret
            }
            true => {
                log::info!("\n\n bmor doing final bmor pass ...");
                // note that state_2 is not saved anywhere, but this last step is easy to do by hans as the caller has the facilties.
                let res = self.bmor_contraction();
                if res.is_err() {
                    std::panic!("bmor_contraction failed");
                }
                //
                let state_2 = res.unwrap();
                state_2.log();
                //
                let facilities = state_2.get_facilities();
                let facilities_ret = facilities.clone();
                facilities_ret
            }
        };
        facilities
    }  // end of end_data


    /// treat data with weights attached.
    /// **This method can be called many times in case of data streaming, passing data by blocks**.  
    /// It returns the number of facilities created up to this call.
    /// a data trplet consists in a weight , data vector and data id  (anything identifying data point)
    pub fn process_weighted_data(&self, weighted_data : &[(f64, &Vec<T>, DataId)] ) -> anyhow::Result<usize>  {
        //
        self.process_weighted_block(weighted_data);
        //
        let state = self.state.borrow();
        //
        state.log();
        if log::log_enabled!(log::Level::Debug) {
            state.get_facilities().log(1);
        }
        //
        return Ok(state.get_facilities().len());
    } // end of process_weighted_data



    // We recur (once) to reduce number of facilities. To go from $1 + k * logn$ to $1 + k * log(log(n))$
    // (We tried to reduce with imp algo but not better)
    pub(crate) fn bmor_contraction(&self) -> anyhow::Result<BmorState<DataId, T, Dist>> {
        //
        log::info!("\n bmor recurring");
        // extract weighted data
        let facility_data = self.state.borrow().get_facilities().into_weighted_data();
        //
        // allocate another Bmor state. TODO: change some parameters gamma ? 
        //
        log::info!("bmor_recur , nb facilities received : {:?}", facility_data.len());
        //
        let weighted_data: Vec<(f64, &Vec<T>, DataId)> = (0..facility_data.len()).into_iter().map( |i| (facility_data[i].0,&facility_data[i].1, facility_data[i].2.clone())).collect();
        let _bound_2 = self.nbdata_expected.ilog2() as usize;
        // we could to adapt to number of facilities and we could impose a log reduction in input size for each step.
        // by bounding nb_expected_data with min(_bound_2)
        // let nb_expected_data = weighted_data.len().min(_bound_2);
        let nb_expected_data = weighted_data.len();
        if self.state.borrow().get_nb_inserted() > self.k * (1 + nb_expected_data.ilog2() as usize) {
            log::debug!("reducing number of facilities: setting expected nb data : {:?}", nb_expected_data);
            let bmor_algo_2 : Bmor<DataId, T, Dist> = Bmor::new(self.get_k(), nb_expected_data , self.get_beta(), self.get_gamma(), 
                                    self.distance.clone());
            //
            let res = bmor_algo_2.process_weighted_data(&weighted_data);
            if res.is_err() {
                return Err(anyhow!("constraction failed"));
            }
            let state_2 = bmor_algo_2.state.borrow();
            state_2.get_facilities().log(0);
            return Ok(state_2.clone()); 
        }
        else {
            let state = self.state.borrow();
            state.log();
            state.get_facilities().log(0);
            return Ok(state.clone()); 
        }
    } // end of bmor_recur





    // This method is the real working method.
    // It inserts data, update state, and drive recurrence
    // args is a vecotr of triplets (weight, data, data_id) 
    fn process_weighted_block(&self, data : &[(f64, &Vec<T>, DataId)]) {
        //
        log::debug!("entering process_weighted_block, phase : {:?}, nb data : {}", self.state.borrow().get_phase(), data.len());
        //
        for d in data {
            // TODO: now we use rank as rank_id (sufficicent for ordered ids)
            log::trace!("treating rank_id : {:?}, weight : {:.4e}", d.2, d.0);
            let add_res = self.add_data(d.2.clone(), &d.1, d.0);
            if !add_res {
                // allocate new state
                log::debug!("recycling facilities, incrementing upper bound for cost, nb_facilities : {:?}", self.state.borrow().get_facilities().len());
                // recycle facilitites in process adding them
                let weighted_data : Vec<(f64,Vec<T>, DataId)>;
                weighted_data = self.state.borrow().centers.get_vec().iter().map(|f| (f.read().get_weight(), f.read().get_position().clone(), f.read().get_dataid())).collect();
                assert!(weighted_data.len() > 0);
                let weighted_ref_data : Vec<(f64,&Vec<T>, DataId)> = weighted_data.iter().map(|wd| (wd.0, &wd.1, wd.2.clone())  ).collect();
                assert!(weighted_ref_data.len() > 0);
                self.state.borrow_mut().reinit(self.beta);
                self.process_weighted_block(&weighted_ref_data);
            }
        }
    } // end of process_weighted_block


    // This function return true except if we got beyond bound for cost or number of facilities
    // The data added can be a facility extracted during a preceding phase
    pub(crate) fn add_data(&self, rank_id : DataId, data : &Vec<T>, weight : f64) -> bool {
        //
        let mut state = self.state.borrow_mut();
        let facilities = state.get_mut_facilities();
        // get nearest facility or open facility
        if facilities.len() <= 0 {
            log::debug!("Bmor::add_data creating facility rank_id : {:?} with weight : {:.3e}", rank_id, weight);
            let mut new_f = Facility::<DataId, T>::new(rank_id, data);
            new_f.insert(weight, 0.);
            facilities.insert(new_f);
            // we update global state here in facility creation case
            state.nb_inserted += 1;
            state.absolute_weight += weight as f64;
            return true;
        }
        // we already have a facility we update state
        let status = state.update(rank_id, data, weight);
        //
        return status;
    } // end of add_data

    pub fn log(&self) {
        self.state.borrow().log();
    }
} // end of impl block Bmor
