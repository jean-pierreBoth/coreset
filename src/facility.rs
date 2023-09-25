//! implement facility management

use anyhow::*;

use ndarray::Array2;
use quantiles::ckms::CKMS;     // we could use also greenwald_khanna

use parking_lot::RwLock;
use std::sync::Arc;

use std::collections::HashMap;

use hnsw_rs::dist::*;

/// A facility is a a center (or point in data) that correspond to a k medoid point
/// The struture stores the data point which serve as a center, the sum of points weight 
/// attached to this point and the cost (distance between data points and center multiplied by point's weight)
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

    pub fn new(d_rank : usize, center : &[T]) -> Self {
        Facility{d_rank,center : center.to_vec(), weight : 0., cost : 0.}
    }

    pub fn get_position(&self) -> &Vec<T> {
        return &self.center;
    }

    /// get data rank this facility is centered on
    pub fn get_dataid(&self) -> usize {
        self.d_rank
    }

    /// return sum of points' weight dipatched to this center
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    pub(crate) fn insert(&mut self, weight : f32, dist : f32) {
        self.weight += weight;
        self.cost += dist * weight;
    }

    pub fn log(&self) {
        log::info!("facility , d_rank : {:?}  weight : {:.5e},  cost : {:.3e}  cost/weight : {:5e}", self.d_rank, self.weight, self.cost, self.cost/self.weight);
    }
} // end of block Facility


//===================================================================================


/// describes the list of facility (or centers created)
/// As we want parallel access, running concurrently on all data we need Arc RwLock stuff
#[derive(Clone)]
pub struct Facilities<T : Send+Sync+Clone, Dist : Distance<T> > {
    centers : Vec<Arc<RwLock<Facility<T>>>>,
    //
    distance : Dist,
}

impl <T:Send+Sync+Clone, Dist : Distance<T> > Facilities<T, Dist> {

    /// to be allocated , size should be log(nb_data)
    pub fn new(size : usize, distance : Dist) -> Self {
        let centers = Vec::<Arc<RwLock<Facility<T>>>>::with_capacity(size);
        Facilities{centers, distance}
    }

    /// return number of facility
    pub fn len(&self) -> usize {
        return self.centers.len()
    }

    /// total weight already inserted
    pub fn get_weight(&self) -> f32 {
        return self.centers.iter().map(|f| f.read().get_weight()).sum()
    }


    // useful in algorithm bmor when we need to reinitialize
    pub(crate) fn clear(&mut self) {
        self.centers.clear();
    }


    // access to internal representation
    pub(crate) fn get_vec(&self) -> &Vec<Arc<RwLock<Facility<T>>>> {
        &self.centers
    }


    // return true if there is a facility around point at distance less than dmax
    pub fn match_point(&self, point : &Vec<T>, dmax : f32, distance : &Dist) -> bool {
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
        //
        log::debug!(" facility insertion nb facilities : {}", self.centers.len());
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
    } // end of get_cloned_facility


    /// return rank of nearest facility
    pub fn get_nearest_facility(&self, data : &[T]) -> anyhow::Result<(usize, f32)> {
        let mut dist = f32::INFINITY;
        let mut rank_f : i32 = -1;
        if self.centers.len() == 0 {
            return Err(anyhow!("Empty facility"));
        }
        // can be // if many centers
        for i in 0..self.centers.len() {
            let f_i = self.get_facility(i).unwrap().read();
            let center_i = f_i.get_position(); 
            let d_i = self.distance.eval(center_i, data);
            if d_i <= dist {
                dist = d_i;
                rank_f = i as i32;
            }
        }
        //
        return Ok((rank_f as usize, dist));
    } // end of get_nearest_facility

    /// a function to log info on dist and cost inside facilities
    pub fn log(&self) {
        for f in &self.centers {
            f.read().log()
        }
    } // end of log 


    /// If we have labelled data we can store labels counts affected to each facility
    /// This function returns total cost and a vector of counts for eacl label occuring in a Facility
    /// Can be useful to check homogneity or clustering
    pub fn dispatch_labels(&self, data : &Vec<Vec<T>>, labels : &Vec<u8>) -> (f64, Vec::<HashMap<u8, u32>>) {
        //
        assert_eq!(data.len(), labels.len());
        //
        let mut global_cost = 0_f64;
        let nb_facility = self.centers.len();
        let mut label_distribution = Vec::<HashMap<u8, u32>>::with_capacity(nb_facility);
        for _ in 0..nb_facility {
            label_distribution.push(HashMap::<u8, u32>::with_capacity(data.len() / (2* nb_facility)));
        }
        //
        for i in 0..data.len() {
            let rank_dist = self.get_nearest_facility(&data[i]).unwrap();
            global_cost += rank_dist.1 as f64;
            if let Some(count) = label_distribution[rank_dist.0].get_mut(&labels[i]) {
                *count += 1;
            }
            else {
                label_distribution[rank_dist.0].insert(labels[i], 1);
            }
        }
        //
        return (global_cost, label_distribution);
    } // end of dispatch_labels


        // TODO: useful?
    /// computes distances between facility
    pub fn cross_distances(&self, distance : &Dist) {
        let nb_facility = self.centers.len();
        let mut distances = Array2::<f32>::zeros((nb_facility , nb_facility));
        let mut q_dist = CKMS::<f32>::new(0.01);

        for i in 0..nb_facility {
            let f_i = self.get_facility(i).unwrap().read();
            let center_i = f_i.get_position();
            for j in 0..nb_facility {
                if i!=j {
                    let f_j = self.get_facility(j).unwrap().read();
                    distances[[i,j]] = distance.eval(center_i, f_j.get_position());
                    q_dist.insert(distances[[i,j]]);
                }
            }
        }
        //
        log::info!("\n cross facility distances quantiles");

        println!("\n distance quantiles at 0.05 : {:.2e},   0.1 : {:.2e} , 0.5 : {:.2e}, 0.75 :  {:.2e} , 0.9 : {:.2e}", 
        q_dist.query(0.05).unwrap().1, q_dist.query(0.1).unwrap().1, q_dist.query(0.5).unwrap().1,  q_dist.query(0.75).unwrap().1, q_dist.query(0.9).unwrap().1);
        log::debug!("\n cross distances : {:.3e}", distances);
        let threshold =  q_dist.query(0.01).unwrap().1;
        log::info!("mergeable threshold : {:.3e}", threshold);
        let mut nbmerge = 0;
        // search mergeable facilities
        for i in 0..nb_facility {
            for j in 0..nb_facility {
                if i < j && distances[[i,j]] < threshold {
                    log::info!("mergeable facilities : (i,j) : ({:?},{:?}) ", i,j);
                    nbmerge+= 1;
                }
            }
        }
        log::info!("nb merge possible : {:?}", nbmerge);
    } // end of cross_distances

} // end of impl block Facilities
