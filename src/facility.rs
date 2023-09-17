//! implement facility management

use parking_lot::RwLock;
use std::sync::Arc;


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

    pub fn get_position(&self) -> &Vec<T> {
        return &self.center;
    }

    pub fn get_rank(&self) -> usize {
        self.d_rank
    }

    pub fn get_weight(&self) -> f32 {
        self.weight
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
    pub fn get_nb_facility(&self) -> usize {
        return self.centers.len()
    }

    // return true if there is a facility around point at distance less than dmax
    fn match_point<Dist : Distance<T>>(&self, point : &Vec<T>, dmax : f32, distance : &Dist) -> bool {
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
} // end of impl block Facilities
