//! implement facility management

use parking_lot::RwLock;
use std::sync::Arc;


use hnsw_rs::dist::*;

#[derive(Clone)]
pub struct Facility<T: Send+Sync + Clone> {
    // rank in data
    d_rank : usize,
    // facility location
    center : Vec<T>,
    // weight (how many points it represents)
    weight : f32,
    //
    cost : f32,
    //
}

impl<T: Send+Sync+Clone> Facility<T> {

    pub fn new(d_rank : usize, center : &Vec<T>) -> Self {
        Facility{d_rank,center : center.clone(), weight : 0., cost : 0.}
    }

    pub fn get_position(&self) -> &Vec<T> {
        return &self.center;
    }

    /// get data rank this facility is centered on
    pub fn get_dataid(&self) -> usize {
        self.d_rank
    }

    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    pub(crate) fn insert(&mut self, weight : f32, dist : f32) {
        self.weight += weight;
        self.cost += dist * weight;
    }

    pub fn log(&self) {
        log::info!("facility , d_rank : {:?}  weight : {:.5e},  cost : {:.3e}", self.d_rank, self.weight, self.cost);
    }
} // end of block Facility


//===================================================================================


/// describes the list of facility (or centers created)
pub struct Facilities<T : Send+Sync+Clone> {
    centers : Vec<Arc<RwLock<Facility<T>>>>
}

impl <T:Send+Sync+Clone> Facilities<T> {

    /// to be allocated , size should be log(nb_data)
    pub fn new(size : usize) -> Self {
        let centers = Vec::<Arc<RwLock<Facility<T>>>>::with_capacity(size);
        Facilities{centers}
    }

    /// return number of facility
    pub fn len(&self) -> usize {
        return self.centers.len()
    }

    // return true if there is a facility around point at distance less than dmax
    pub fn match_point<Dist : Distance<T>>(&self, point : &Vec<T>, dmax : f32, distance : &Dist) -> bool {
        //
        for f in &self.centers {
            if distance.eval(f.read().get_position(), point) <= dmax {
                return true;
            }
        }
        return false;
    } // end of match_facility


    ///
    pub(crate) fn insert(&mut self, facility : Facility<T>) {
        self.centers.push(Arc::new(RwLock::new(facility)));
    }


    /// retrieve facility by rank if rank is Ok
    pub fn get_facility(&self, rank : usize) -> Option<&Arc<RwLock<Facility<T>>>> {
        if rank >= self.centers.len() {
            return None;
        }
        else {
            return Some(&self.centers[rank]);
        }
    }

    /// retrieve facility of given rank , and returns a clone (to avoid synchro stuff)
    /// useful for easy final analysis
    pub fn get_cloned_facility(&self, rank : usize) -> Option<Facility<T>> {
        if rank >= self.centers.len() {
            return None;
        }
        else {
            return Some(self.centers[rank].read().clone());
        }        
    }

} // end of impl block Facilities
