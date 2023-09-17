//! adaptation of Streaming k-means on well clustered data
//! Braverman Meyerson Ostrovski Roytman ACM-SIAM 2011
//! 
//! 
//! 
//! 
use anyhow::{anyhow, Result};

use rayon::prelude::*;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use rand::distributions::{Distribution,Uniform};

use hnsw_rs::dist::*;



#[derive(Clone)]
pub struct Facility<T: Send+Sync> {
    // rank in data
    d_rank : usize,
    // weight (how many points it represents)
    weight : f32,
    // facility location
    center : Vec<T>
}

impl<T: Send+Sync> Facility<T> {

    pub fn new(d_rank : usize, weight : f32, center : Vec<T>) -> Self {
        Facility{d_rank, weight, center}
    }

    pub fn get_center(&self) -> &Vec<T> {
        return &self.center;
    }

    pub fn get_rank(&self) -> usize {
        self.d_rank
    }

    pub fn get_weight(&self) -> f32 {
        self.weight
    }
} // end of block Facility



#[derive(Clone)]
pub struct BmorState<T:Send+Sync> {
    // nb iterations (phases)
    phase : usize,
    // at each phase we have an upper bound for cost.
    phase_cost_upper : f32,
    // current centers, associated to rank in stream (or in data) and weight (nb points in facility)
    centers : Vec<Facility<T>>,
} // end of 


impl<T:Send+Sync> BmorState<T> {

    pub(crate) fn new(phase : usize, alloc_size : usize, upper_cost : f32) -> Self {
        let centers = Vec::<Facility<T>>::with_capacity(alloc_size);
        BmorState{phase, phase_cost_upper : upper_cost, centers}
    }


    pub fn get_centers(&self) -> &Vec<Facility<T>> {
        return &self.centers
    }

    // get current phase num of processing
    pub(crate) fn get_phase(&self) -> usize {
        self.phase
    }

    pub(crate) fn get_phase_cost_bound(&self) -> f32 {
        self.phase_cost_upper
    }

    /// get nearest center/facility of a point
    pub fn get_nearest_center<Dist : Distance<T>> (&self, point : &[T], distance : &Dist) -> Option<&Facility<T>> 
        where T : Send + Sync,  Dist : Sync {
        //
        let nb_facility = self.centers.len();
        //
        if nb_facility == 0 {
            return None;
        }
        // get nearest facilty
        let mut distances : Vec<(usize, f32)> = (0..nb_facility).into_par_iter().map(|i| (i, distance.eval(self.centers[i].get_center(), point))).collect();
        distances.sort_unstable_by(|it1, it2| it1.1.partial_cmp(&it2.1).unwrap());
        let nearest = distances[0];
        //
        return Some(&self.centers[nearest.0]);
    }

    // reinitialization. (upper cost rescaling)
    pub(crate) fn reinit(&mut self, phase_cost_upper : f32) {
        self.phase += 1;
        self.phase_cost_upper = phase_cost_upper;
        self.centers.clear();
    }

} // end of impl block BmorState





pub struct Bmor {
    // base number of centers expected
    k : usize,
    //
    nbdata_expected : usize,
    // cost multiplicative factor for upper bound of accepted cost at each phase.
    beta : f32,
    //  slackness parameters for cost and number of centers accepted
    gamma : f32,
    //
    f_scale : f64,
}  // end of struct Bmor



impl Bmor {

    /// - k: number of centers
    /// - nbdata : nb data expected,
    /// - gamma 
    pub fn new(k: usize, nbdata : usize, beta : f32) -> Self {
        let gamma = 10.;
        // TODO: to be adapted?
        let phase_cost_upper = 1.;
        let f_scale = phase_cost_upper/ (k as f32 * (1. + nbdata.ilog2() as f32));
        Bmor{k, nbdata_expected : nbdata, beta, gamma, f_scale : f_scale.into()}
    }


    pub fn process_block<T : Send + Sync + Clone>(&mut self, data : &Vec<Vec<T>>) {
        //
        let mut nb_centers_bound = self.gamma * (1. + self.nbdata_expected.ilog2() as f32) * self.k as f32; 
        let upper_cost = 1.;
        let mut state = BmorState::<T>::new(0, nb_centers_bound as usize, upper_cost);
        //
        let weighted_data: Vec<(f32, &Vec<T>)> = data.iter().map( |v| (1.,v)).collect();
        self.process_weighted_block(&mut state, &weighted_data);
    } // end of process_block



    // This method can do block processing as dispatched by 
    // recurring processing
    pub fn process_weighted_block<T : Send + Sync + Clone>(&mut self, state : &mut BmorState<T>, data : &Vec<(f32,&Vec<T>)>) {
        //
        log::debug!("entering process_weighted_block, pahse : {:?}", state.get_phase());
        //
        for d in data {
            let add_res = self.add_data(state, &d.1, d.0);
            if !add_res {
                // allocate new state
                log::debug!("merging facilities, incrementing upper bound for cost, nb_facilities : {:?}", state.get_centers().len());
                let old_state = state.clone();
                state.reinit(self.beta * old_state.get_phase_cost_bound());
                // recycle facilitites in process addind them
                let weighted_data : Vec<(f32,Vec<T>)>;
                weighted_data  =  state.centers.iter().map(|f| (f.get_weight(), f.get_center().clone())).collect();
                let weighted_ref_data : Vec<(f32,&Vec<T>)> = weighted_data.iter().map(|wd| (wd.0, &wd.1)  ).collect();
                self.process_weighted_block(state, &weighted_ref_data);
            }
        }
    } // end of process_weighted_block


    // This function return true except if we got beyond bound for cost or number of facilities
    pub fn add_data<T: Send+ Sync>(&self, state : &mut BmorState<T>, data : &Vec<T>, weight : f32) -> bool {
        // get nearest facility or open facility
        
        // do we break bound on cost or facility number

        return true;
    }


} // end of impl block Bmor
