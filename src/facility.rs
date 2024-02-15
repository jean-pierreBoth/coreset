//! implement facility management.  
//! Each facility maintain weights of data items dispatched to it and cost contribution
//!

use anyhow::*;

use serde::{Deserialize, Serialize};

use ndarray::Array2;
use quantiles::ckms::CKMS; // we could use also greenwald_khanna

use rayon::prelude::*;

use parking_lot::RwLock;
use std::sync::Arc;

use std::collections::HashMap;

use hnsw_rs::dist::*;

/// A facility is a dataid and the center (or point in data) that correspond to a k medoid point.  
/// The struture stores the data vector and point id which serve as a center, the sum of points weight
/// attached to this point and the cost (distance between data points and center multiplied by point's weight)
#[derive(Clone, Serialize, Deserialize)]
pub struct Facility<DataId, T: Send + Sync + Clone> {
    // rank in data
    d_rank: DataId,
    // facility location
    center: Vec<T>,
    // weight (how many points it represents)
    weight: f64,
    //
    cost: f64,
    //
} // end of Facility

impl<DataId: std::fmt::Debug + Clone, T: Send + Sync + Clone> Facility<DataId, T> {
    /// creates a facility, around a point characteristics,
    /// As the point could be different from data as in kmean we do not set weight.
    /// So an explicit insertion with method insert must be done when creation facility is
    /// mean to also insert
    pub fn new(d_rank: DataId, center: &[T]) -> Self {
        Facility {
            d_rank,
            center: center.to_vec(),
            weight: 0.,
            cost: 0.,
        }
    }

    /// get a data point corresponding to the facility location
    pub fn get_position(&self) -> &Vec<T> {
        return &self.center;
    }

    /// get data rank this facility is centered on
    pub fn get_dataid(&self) -> DataId {
        self.d_rank.clone()
    }

    /// return sum of points' weight dipatched to this center
    pub fn get_weight(&self) -> f64 {
        self.weight
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// return cost carried by this facility $f$ i.e :  $ cost(f) = \sum_{p \in f} w(p) * dist(p,f) $
    pub fn get_cost(&self) -> f64 {
        self.cost
    }

    // This function increments weight and cost related to a facility
    pub(crate) fn insert(&mut self, weight: f64, dist: f32) {
        self.weight += weight;
        self.cost += dist as f64 * weight;
    }

    // This function empties a facility keeping its position
    pub(crate) fn empty(&mut self) {
        self.weight = 0.;
        self.cost = 0.;
    }

    /// dumps weight, cost and ratio
    pub fn log(&self) {
        log::info!(
            "facility , dataid : {:?}  weight : {:.4e},  cost : {:.3e}  cost/weight : {:.3e}",
            self.d_rank,
            self.weight,
            self.cost,
            self.cost / self.weight
        );
    }
} // end of block Facility

//===================================================================================

#[cfg_attr(doc, katexit::katexit)]
/// Describes the list of facility (or centers created).
///  
/// The structure maintains the list of open facilities (or cluster) and their centers.
/// It computes and store (see [compute_weight_cost](compute_weight_cost)) total weight dispatched into facilities and maintain
/// global facility cost assignment  as :
///   $ \sum_{p \in P}  \medspace  w(p) * dist(p, cf_{p})$.
/// where $ cf_{p}$ is the center of facility assigned to $p$
///
// As we want parallel access, running concurrently on all data we need Arc RwLock stuff

pub type FacilityId = usize;

/// A structure describing point affectation into facility
/// Necessary to comput sensitiviy in coreset algos
#[derive(Copy, Clone, Debug)]
pub struct PointMap {
    // rank of facility corresponding to point
    facility: usize,
    // distance to facility
    dist_to_f: f32,
    // point weight
    weight: f32,
}

impl PointMap {
    pub fn new(facility: usize, dist_to_f: f32, weight: f32) -> Self {
        PointMap {
            facility,
            dist_to_f,
            weight,
        }
    }

    /// returns facility of point
    pub fn get_facility(&self) -> usize {
        self.facility
    }

    //
    pub fn get_dist(&self) -> f32 {
        self.dist_to_f
    }

    // get point weight
    pub fn get_weight(&self) -> f32 {
        self.weight
    }
} // end of PointMap

/// This structuree represents the list of facilities created
#[derive(Clone)]
pub struct Facilities<DataId, T: Send + Sync + Clone, Dist: Distance<T>> {
    centers: Vec<Arc<RwLock<Facility<DataId, T>>>>,
    //
    distance: Dist,
    // sum of weights dispatched into facilities
    weight: f64,
    // sum of weights * distance to facility center dispatched into facilities
    cost: f64,
} // end of struct Facilities

impl<
        DataId: std::fmt::Debug + Clone + Send + Sync,
        T: Send + Sync + Clone,
        Dist: Distance<T> + Send + Sync,
    > Facilities<DataId, T, Dist>
{
    /// to be allocated , size should be log(nb_data)
    pub fn new(size: usize, distance: Dist) -> Self {
        let centers = Vec::<Arc<RwLock<Facility<DataId, T>>>>::with_capacity(size);
        Facilities {
            centers,
            distance,
            weight: 0.,
            cost: 0.,
        }
    }

    /// return number of facility
    pub fn len(&self) -> usize {
        return self.centers.len();
    }

    /// total weight already inserted
    pub fn get_weight(&self) -> f64 {
        return self.centers.iter().map(|f| f.read().get_weight()).sum();
    }

    pub fn get_distance(&self) -> &Dist {
        &self.distance
    }

    /// returns sum of costs dispatched into facilities.
    pub fn get_cost(&self) -> f64 {
        return self.centers.iter().map(|f| f.read().get_cost()).sum();
    }

    // deletes all facilities. useful in algorithm bmor when we need to reinitialize.
    pub(crate) fn clear(&mut self) {
        log::debug!("clearing facilities");
        self.centers.clear();
        self.weight = 0.;
        self.cost = 0.;
    }

    // keeps facilities but empty each of them. Enables new dispatching of points in facilities
    // useful in coreset construction
    pub(crate) fn empty(&mut self) {
        log::debug!("emptying facilities");
        for f in &self.centers {
            f.write().empty();
        }
        self.weight = 0.;
        self.cost = 0.;
    }

    // access to internal representation
    pub(crate) fn get_vec(&self) -> &Vec<Arc<RwLock<Facility<DataId, T>>>> {
        &self.centers
    }

    /// return true if there is a facility around point at distance less than dmax
    pub fn match_point(&self, point: &Vec<T>, dmax: f32, distance: &Dist) -> bool {
        //
        for f in &self.centers {
            if distance.eval(f.read().get_position(), point) <= dmax {
                return true;
            }
        }
        return false;
    } // end of match_facility

    /// insert a new facility
    pub(crate) fn insert(&mut self, facility: Facility<DataId, T>) {
        self.centers.push(Arc::new(RwLock::new(facility)));
        //
        log::trace!(
            "Facilities: facility insertion nb facilities : {}",
            self.centers.len()
        );
    }

    /// retrieve facility by rank if rank is Ok
    pub fn get_facility(&self, rank: usize) -> Option<&Arc<RwLock<Facility<DataId, T>>>> {
        if rank >= self.centers.len() {
            return None;
        } else {
            return Some(&self.centers[rank]);
        }
    }

    /// retrieve facility of given rank , and returns a clone (to avoid synchro stuff)
    /// useful for easy final analysis
    pub fn get_cloned_facility(&self, rank: usize) -> Option<Facility<DataId, T>> {
        if rank >= self.centers.len() {
            return None;
        } else {
            return Some(self.centers[rank].read().clone());
        }
    } // end of get_cloned_facility

    /// return weight in facility of rank rank, error else
    pub fn get_facility_weight(&self, rank: usize) -> Result<f64> {
        if rank <= self.centers.len() {
            return Ok(self.centers[rank].read().get_weight());
        } else {
            return Err(anyhow!("not so many facilities , rank is {}", rank));
        }
    } // end of facililities_ref

    /// return rank of nearest facility and distance to it
    /// If there are many facilities to search (thousands), setting the parallel flag to true is useful
    pub fn get_nearest_facility(&self, data: &[T], parallel: bool) -> anyhow::Result<(usize, f32)> {
        let mut dist = f32::INFINITY;
        let mut rank_f: usize = usize::MAX;
        if self.centers.len() == 0 {
            return Err(anyhow!("Empty facility"));
        }
        //
        let dist_to_f = |i| -> (usize, f32) {
            let f_i = self.get_facility(i).unwrap().read();
            let center_i = f_i.get_position();
            let d_i = self.distance.eval(center_i, data);
            (i, d_i)
        };
        let dist_slot: Vec<(usize, f32)> = match parallel {
            true => (0..self.centers.len())
                .into_par_iter()
                .map(|i| dist_to_f(i))
                .collect(),
            false => (0..self.centers.len())
                .into_iter()
                .map(|i| dist_to_f(i))
                .collect(),
        };
        for (f, d) in dist_slot {
            if d < dist {
                dist = d;
                rank_f = f;
            }
        }
        assert!(rank_f < usize::MAX);
        //
        return Ok((rank_f as usize, dist));
    } // end of get_nearest_facility

    /// insert a point into given facility (must be the one given by get_nearest_facility)
    pub(crate) fn insert_point(&self, facility: usize, dist: f32, weight: f32) {
        let mut f = self.centers[facility].write();
        f.weight += weight as f64;
        f.cost += dist as f64 * weight as f64;
    }

    /// returns (total weight, total cost)
    ///
    pub fn compute_weight_cost(&mut self) -> (f64, f64) {
        //
        if self.weight <= 0. {
            let mut total_weight = 0.;
            let mut total_cost = 0.;
            for f in &self.centers {
                let f_access = f.read();
                total_cost += f_access.get_cost();
                total_weight += f_access.get_weight();
            }
            self.cost = total_cost;
            self.weight = total_weight;
        }
        return (self.weight, self.cost);
    } // end of compute_cost

    /// a function to log info on dist and cost inside facilities
    /// - level = 0, will log total weight and total cost summed over facilities
    /// - level = 1 it will log weight and cost of each facility.
    pub fn log(&self, level: usize) {
        let mut total_weight = 0.;
        let mut total_cost = 0.;
        for f in &self.centers {
            let f_access = f.read();
            if level == 1 {
                f_access.log();
            }
            total_cost += f_access.get_cost();
            total_weight += f_access.get_weight();
        }
        log::info!(
            "\n\n sum of facilities weight : {:.3e}, total cost : {:.3e}",
            total_weight,
            total_cost
        );
        log::info!("nb facilities : {}", self.centers.len());
        log::info!("\n *************************************************");
    } // end of log

    /// This function affects each point to its nearest facility and compute cost of each facility and returns global cost and vector of weight by facility
    /// arguments are data vectors, data ids and optional weights.  
    ///   
    /// Not all algos maintain weight and cost consistently all the way, sometimes facilities are created without
    /// searching all points in it. So we need to be able do the dispatch afterwards.  
    /// **Bmor algorithm dispatch points on the fly so it computes an upper bound of the cost**  
    /// **The function can nevertheless be called a posteriori to get a tighter bound on cost**  
    ///
    #[allow(unused)]
    pub fn dispatch_data(
        &mut self,
        data: &Vec<&Vec<T>>,
        ids: &Vec<usize>,
        weights: Option<&Vec<f32>>,
    ) -> f64 {
        //
        log::info!("in facilities::dispatch_data");
        //
        if weights.is_some() {
            assert_eq!(data.len(), weights.unwrap().len());
        }
        // keep facilities but empty facilities keep them at their position
        self.empty();
        //
        let dispatch_i = |item: usize| {
            // get facility rank and weight
            // parallel flag is set to false as we // on data.
            let (facility, dist) = self.get_nearest_facility(&data[item], false).unwrap();
            let weight = if weights.is_none() {
                1.
            } else {
                weights.unwrap()[item]
            };
            self.insert_point(facility, dist, weight);
        };
        //
        (0..data.len())
            .into_par_iter()
            .for_each(|item| dispatch_i(item));
        //
        let mut global_cost = 0_f64;
        let mut total_weight = 0.;
        for i in 0..self.centers.len() {
            global_cost += self.centers[i].read().cost;
            total_weight += self.centers[i].read().weight;
        }
        //
        println!(
            "\n\n total weight collected in facilities : {:.3e}, total cost : {:.3e}",
            total_weight, global_cost
        );
        println!("\n **************************************************************************");
        //
        global_cost
    } // end of dispatch_data

    /// If we have labelled data we can store labels counts affected to each facility.  
    /// This function dispatch **data and labels** into facilities and returns total cost and a vector of counts for each label occuring in a Facility.  
    /// It computes for each facililty label distribution, entropy of distribution and can be used to check clustering.
    /// **This methods can be called after processing all the data**.     
    /// Returns Vector of label distribution entropy by facility and distribution as a HashMap
    pub fn dispatch_labels<L: PartialEq + Eq + Copy + std::hash::Hash + Sync + Send>(
        &mut self,
        data: &Vec<Vec<T>>,
        labels: &Vec<L>,
        weights: Option<&Vec<f32>>,
    ) -> (Vec<f64>, Vec<HashMap<L, u32>>) {
        //
        log::info!("dispatch_labels");
        //
        type SafeHashMap<L> = Arc<RwLock<HashMap<L, u32>>>;
        assert_eq!(data.len(), labels.len());
        //
        let nb_facility = self.centers.len();
        let mut label_distribution = Vec::<SafeHashMap<L>>::with_capacity(nb_facility);

        for i in 0..nb_facility {
            // reinitialize weights and cost of facilities
            self.centers[i].write().cost = 0.;
            self.centers[i].write().weight = 0.;
            // allocate hashmaps
            let newmap = HashMap::<L, u32>::with_capacity(data.len() / (2 * nb_facility));
            label_distribution.push(Arc::new(RwLock::new(newmap)));
        }
        //
        let dispatch_i = |i: usize| {
            // find facility
            let (itemf, dist) = self.get_nearest_facility(&data[i], false).unwrap();
            // dispatch data
            let weight = if weights.is_none() {
                1.
            } else {
                weights.unwrap()[itemf] as f64
            };
            let cost_incr = dist as f64 * weight;
            {
                let mut facility = self.centers[itemf].write();
                facility.weight += weight;
                facility.cost += cost_incr;
            }
            // dispatch label
            {
                // write lock
                let mut distribution = label_distribution[itemf].write();
                if let Some(count) = distribution.get_mut(&labels[i]) {
                    *count += 1;
                } else {
                    distribution.insert(labels[i], 1);
                }
            }
        };
        //
        log::info!("computing global cost and weights");
        (0..data.len())
            .into_par_iter()
            .for_each(|item| dispatch_i(item));
        // recompute globlas cost and weight
        let mut global_cost = 0_f64;
        let mut total_weight = 0.;
        for i in 0..nb_facility {
            global_cost += self.centers[i].read().cost;
            total_weight += self.centers[i].read().weight;
        }
        //
        println!(
            "\n\n total weight collected in facilities : {:.3e}, total cost : {:.3e}",
            total_weight, global_cost
        );
        println!("\n **************************************************************************");
        //
        // We can compute entropy distribution
        //
        log::info!("computing label distribution entropy");
        let mut entropies = Vec::<f64>::with_capacity(nb_facility);
        for i in 0..nb_facility {
            let distribution = label_distribution[i].read();
            let mut mass = 0.0f64;
            let nb_label = distribution.len();
            let mut weights = Vec::<f64>::with_capacity(nb_label);
            let mut entropy = 0.;
            for item in distribution.iter() {
                assert!(*item.1 > 0);
                weights.push(*item.1 as f64);
                mass += *item.1 as f64;
                entropy -= (*item.1 as f64) * (*item.1 as f64).ln();
            }
            entropy = entropy / mass + mass.ln();
            // checks
            if entropy < -f64::EPSILON * 10. {
                log::error!("facility {:?} entropy {:.3e}", i, entropy);
                std::panic!("negative entropy");
            } else {
                entropy = entropy.max(0.);
            }
            entropies.push(entropy);
        }
        // Construct global weighted entropy measure
        let mut global_entropy = 0.;
        let mut total_weight = 0.;
        for i in 0..nb_facility {
            let facility = self.centers[i].read();
            let weight = facility.get_weight();
            total_weight += weight;
            global_entropy += weight * entropies[i];
        }
        global_entropy /= total_weight;
        println!(
            "\n\n mean of entropies : {:.3e}, total weight : {:.3e}",
            global_entropy, total_weight
        );
        println!("\n **************************************************************************");
        //
        let mut simple_label_distribution = Vec::<HashMap<L, u32>>::with_capacity(nb_facility);
        for i in 0..nb_facility {
            simple_label_distribution.push(label_distribution[i].read().clone());
        }
        //
        return (entropies, simple_label_distribution);
    } // end of dispatch_labels

    /// extract facility centers and associated weight for possible other clustering step
    pub fn into_weighted_data(&self) -> Vec<(f64, Vec<T>, DataId)> {
        log::info!("facility::into_weighted_data");
        //
        let nb_facility = self.len();
        let mut weighted_data = Vec::<(f64, Vec<T>, DataId)>::with_capacity(nb_facility);
        for i in 0..nb_facility {
            let facility = self.get_facility(i).unwrap().read();
            let weight = facility.get_weight();
            let pos = facility.get_position();
            let id: DataId = facility.get_dataid();
            weighted_data.push((weight, pos.clone(), id.clone()));
        }
        weighted_data
    } // end of into_weighted_data

    /// computes distances between facility
    pub fn cross_distances(&self) {
        let nb_facility = self.centers.len();
        let mut distances = Array2::<f32>::zeros((nb_facility, nb_facility));
        let mut q_dist = CKMS::<f32>::new(0.01);

        if nb_facility <= 1 {
            log::error!("facility::cross_distances, only one facility");
            return;
        }
        for i in 0..nb_facility {
            let f_i = self.get_facility(i).unwrap().read();
            let center_i = f_i.get_position();
            for j in 0..nb_facility {
                if i != j {
                    let f_j = self.get_facility(j).unwrap().read();
                    distances[[i, j]] = self.distance.eval(center_i, f_j.get_position());
                    q_dist.insert(distances[[i, j]]);
                }
            }
        }
        //
        println!("\n inter facility distances quantiles : ");
        println!("\n distance quantiles at  0.01 : {:.2e}, 0.05 :  {:.2e},   0.1 : {:.2e} , 0.5 : {:.2e}, 0.75 :  {:.2e} ", 
        q_dist.query(0.01).unwrap().1, q_dist.query(0.05).unwrap().1, q_dist.query(0.1).unwrap().1, q_dist.query(0.5).unwrap().1,  q_dist.query(0.75).unwrap().1);
        //
        log::debug!("\n cross distances : {:.3e}", distances);
    } // end of cross_distances
} // end of impl block Facilities
