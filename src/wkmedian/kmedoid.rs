//! Implementation of Kmeds in
//!     - Park-Jun Simple and Fast algorithm for k-medoids clustering 2009
//!     with :
//!       - introduction of weights attached to data beccause coreset data have a weight
//!       - cost a point is its weight multiplied by distance to centers
//!       - initialization of medoids is done by decreasing costs  
//!
//!  This mimics the kmean algo. The weights attached to points alleviates the problem of local minima.
//!  See also Friedmann Hastie Tibshirani, The Elements Of Statistical Learning 2001 (Clustering chapter)
//!
//!
//!

use ndarray::Array2;

use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;


use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use quantiles::ckms::CKMS;
use rayon::iter::{IntoParallelIterator, ParallelIterator}; // we could use also greenwald_khanna

use hnsw_rs::dist::*;

use crate::makeiter::*;
use crate::sensitivity::*;

// maintain center and cost of each cluster
struct CenterCost(Vec<(usize, f32)>);

// maintain membership and distance to its center for each point
struct MemberDist(Vec<(u32, f32)>);

#[derive(Copy, Clone)]
pub struct Medoid<DataId> {
    /// id as given by coreset
    center_id: DataId,
    /// rank of points in ids of points as given in struct Coreset
    center: u32,
    /// cost of thi cluster
    cost: f32,
}

impl<DataId> Medoid<DataId>
where
    DataId: Default + Clone,
{
    //
    fn default() -> Self {
        Medoid {
            center_id: DataId::default(),
            center: u32::MAX,
            cost: f32::MAX,
        }
    }

    fn new(id: DataId, center: u32, cost: f32) -> Self {
        Medoid {
            center_id: id,
            center,
            cost,
        }
    }

    pub fn get_center_id(&self) -> DataId {
        self.center_id.clone()
    }

    /// get center (rank)
    fn get_center(&self) -> u32 {
        self.center
    }

    pub fn get_cost(&self) -> f32 {
        self.cost
    }

    fn set_cost(&mut self, cost: f32) {
        self.cost = cost;
    }
}

//TODO: add field from id to rank
/// This algorithm stores the whole matrix distance between points as coreset must have reduced the number of points to a few thousands.
pub struct Kmedoid<DataId, T> {
    //
    nb_cluster: usize,
    // orginal ids of data to cluster i.e those in the coreset (!!) by line of matrix
    ids: Vec<DataId>,
    // distance matrix
    distance: Array2<f32>,
    // weights of points in coreset in order corresponding to lines of distance matrix
    weights: Vec<f64>,
    // current affectation of each coreset point. For each point returns rank in medoids array.
    membership: Vec<u32>,
    // medoid : a Vector containing the nb_cluster medoid
    medoids: Vec<Medoid<DataId>>,
    // at end end of computations we keep just the data of the Medoid centers.
    // It is not stored in each medoid for dispatching computing efficacitty
    // Storing is in the same order as in field medoids!!
    centers: Option<Vec<Vec<T>>>,
    //
    d_quantiles: CKMS<f32>,
} // end of struct Kmedoid

impl<DataId, T> Kmedoid<DataId, T>
where
    DataId: Eq + std::hash::Hash + Send + Sync + Clone + Default,
    T: Send + Sync + Clone,
{
    pub fn new<Dist>(coreset: &CoreSet<DataId, T, Dist>, nb_cluster: usize) -> Self
    where
        Dist: Distance<T> + Send + Sync + Clone,
    {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        let (ids, distance) = coreset.compute_distances().unwrap();
        log::info!(
            "\n ======kmedoids  distance matrix init sys time(ms) {:?} cpu time(ms) {:?}\n ",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        //
        let nbpoints = coreset.get_nb_points();
        log::info!("Kmedoid received coreset of size : {}", nbpoints);
        // weight[i] corresponds to row[i] in distance matrix. From now on all computation use center as raks and no ids.
        let mut weights = Vec::<f64>::with_capacity(nbpoints);
        for i in 0..ids.len() {
            let weight = coreset.get_weight(ids[i].clone()).unwrap();
            weights.push(weight);
        }
        //
        assert_eq!(weights.len(), nbpoints);
        //
        let membership = (0..nbpoints).into_iter().map(|_| u32::MAX).collect();
        let medoids = (0..nb_cluster)
            .into_iter()
            .map(|_| Medoid::default())
            .collect();
        //
        //
        return Kmedoid {
            nb_cluster,
            ids,
            distance,
            weights,
            membership,
            medoids,
            centers: None,
            d_quantiles: CKMS::<f32>::new(0.01),
        };
    } // end of new

    /// nb_iter is maximal number of iterations  
    /// returns best result as couple (iteration, cost)
    pub fn compute_medians(&mut self, nb_iter : usize) -> (usize, f32) {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        log::info!("\n\nentering Kmedoid::Kmedoid");
        log::info!("==============================");
        self.d_quantiles = self.quantile_estimator();
        log::info!(
            "======kmedoids  quantiles done sys time(ms) {:?} cpu time(ms) {:?}\n ",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        let mut perturbation_set = Vec::<(usize, usize)>::new();
        //
        // initialize: random selection of centers, dispatch points to nearest centers
        //
        let mut centers = self.max_cost_init(); // select nb_cluster different points
        log::info!(
            "======kmedoids  center init done sys time(ms) {:?} cpu time(ms) {:?}\n ",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        //
        let mut medoids: Vec<Medoid<DataId>> = centers
            .iter()
            .map(|i| Medoid::new(self.ids[*i as usize].clone(), *i, f32::MAX))
            .collect();
        self.medoids = medoids.clone();
        //
        // we set initial state
        // dispatch each point to nearest center, i.e set membership.
        //
        let mut membership_and_dist = self.dispatch_to_medoids(&centers);
        assert_eq!(self.ids.len(), membership_and_dist.0.len());
        for i in 0..membership_and_dist.0.len() {
            self.membership[i] = membership_and_dist.0[i].0;
        }
        let costs = self.compute_medoids_cost(&membership_and_dist); // compute_medoid_cost not called any more after that
        for i in 0..self.medoids.len() {
            self.medoids[i].set_cost(costs[i]);
            self.medoids[i].center = centers[i];
        }
        // we have centers and cost
        let mut monitoring: Vec<(usize, f32)> = Vec::with_capacity(25);
        let mut last_cost = costs.iter().sum::<f32>();
        monitoring.push((0, last_cost));
        let mut best_iter = (0, last_cost);
        //
        log::info!("medoids initialized , global cost : {:.3e}", last_cost);
        log::info!(
            "======kmedoid medoids initialized  sys time(ms) {:?} cpu time(ms) {:?}\n ",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        //
        // iterate
        //
        let mut perturbation = false;
        let mut iteration = 0;
        loop {
            // recompute centers from membership and update clusters costs:  Cpu cost is here
            let centers_and_costs = self.from_membership_to_centers(&membership_and_dist);
            assert_eq!(self.nb_cluster, centers_and_costs.0.len());
            for i in 0..centers_and_costs.0.len() {
                centers[i] = centers_and_costs.0[i].0 as u32
            }
            // compute global cost
            let iter_cost: f32 = centers_and_costs.0.iter().map(|x| x.1).sum();
            //
            if iter_cost >= last_cost && !perturbation {
                log::info!("iteration got a local minimum : {}", iteration);
                let res = self.quality_summary(&perturbation_set, false);
                if res.is_some() {
                    let couple = res.unwrap();
                    if couple.0 < couple.1 {
                        perturbation_set.push(couple);
                    }
                    else {
                        perturbation_set.push((couple.1, couple.0));
                    }
                    for i in 0..medoids.len() {
                        medoids[i] = self.medoids[i].clone();
                    }
                    perturbation = self.center_perturbation(couple, &mut medoids);
                    if perturbation {
                        log::info!("perturbated couple : {:?}", couple);
                        for i in 0..medoids.len() {
                            centers[i] = medoids[i].get_center();
                        }
                        membership_and_dist = self.dispatch_to_medoids(&centers);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                perturbation = false;
                last_cost = iter_cost;
                log::info!("iteration {}, global cost : {:.3e}", iteration, last_cost);
                // we must store our best state
                if iter_cost < best_iter.1 {
                    best_iter = (iteration, iter_cost);
                    self.store_state(&centers_and_costs, &membership_and_dist);
                }
                // we must compute distance to new centers and reassign membership
                membership_and_dist = self.dispatch_to_medoids(&centers);
                assert_eq!(membership_and_dist.0.len(), self.membership.len());
                iteration += 1;
                if iteration >= nb_iter {
                    log::info!("exiting after nb iteration : {}", iteration);
                    break;
                }
            }
        }
        //
        log::info!("======================================================");
        log::info!("best iter : {}, cost {:.3e}", best_iter.0, best_iter.1);
        log::info!("======================================================");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            "kmedoid compute medians total time sys time(ms) {:?} cpu time(ms) {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        //
        self.quality_summary(&perturbation_set, true);
        //
        best_iter
    } // end of compute_medians

    /// stores data vectors for each cluster
    pub(crate) fn retrieve_cluster_centers<IterProducer>(&mut self, iter_producer: &IterProducer)
    where
        IterProducer: MakeIter<DataId = DataId, DataType = Vec<T>>,
    {
        //
        // get a list of ids to find, then scan data
        //
        if self.centers.is_some() {
            log::error!("Kmedoid::retrieve_cluster_centers : centers have already been retrived");
            return;
        }
        let mut data_iter = iter_producer.makeiter();
        let centers_ids: Vec<DataId> = self.medoids.iter().map(|m| m.get_center_id()).collect();
        let mut centers_data = vec![Vec::<T>::new(); centers_ids.len()];
        //
        let mut nb_found = 0;
        while let Some((data_id, data)) = data_iter.next() {
            for i in 0..centers_ids.len() {
                if centers_ids[i] == data_id {
                    centers_data[i] = data;
                    nb_found += 1;
                    break;
                }
            }
            if nb_found == centers_ids.len() {
                break;
            }
        } // end on data
          //
        assert!(nb_found == centers_ids.len());
        //
        self.centers = Some(centers_data);
    } // end of

    fn store_state(&mut self, centers_and_costs: &CenterCost, membership_and_dist: &MemberDist) {
        //
        assert_eq!(centers_and_costs.0.len(), self.medoids.len());
        //
        for i in 0..self.medoids.len() {
            // we do not update center_id we do not use it, we update only at end
            self.medoids[i].center = centers_and_costs.0[i].0 as u32;
            self.medoids[i].center_id = self.ids[centers_and_costs.0[i].0].clone();
            self.medoids[i].cost = centers_and_costs.0[i].1;
        }
        for i in 0..membership_and_dist.0.len() {
            self.membership[i] = membership_and_dist.0[i].0;
        }
    } // end of store_state

    /// return Medoids
    pub fn get_clusters(&self) -> &Vec<Medoid<DataId>> {
        &self.medoids
    }

    /// returns a reference to center of Medoid of rank if centers have already been calculated, None otherwise.
    /// Aborts if rank above len
    pub fn get_cluster_center(&self, rank: usize) -> Option<&Vec<T>> {
        match &self.centers {
            Some(centers) => {
                if rank < centers.len() {
                    Some(&centers[rank])
                } else {
                    log::error!(
                        "Kmedois::get_cluster_center rank asked exceeds len, nb cluster is : {}",
                        centers.len()
                    );
                    std::panic!();
                }
            }
            None => {
                log::error!("Kmedois::get_cluster_center centers not computed yet");
                None
            }
        }
    } // end of get_cluster_center

    /// returns centers. Used for dispatching whole data à posteriori
    pub(crate) fn get_centers(&self) -> Option<&Vec<Vec<T>>> {
        self.centers.as_ref()
    } // end of get_centers

    /// return number of points (to cluster)
    pub fn get_size(&self) -> usize {
        self.ids.len()
    }

    /// returns for each points the rank of its cluster
    pub fn get_membership(&self) -> &Vec<u32> {
        &self.membership
    }

    /// return global partition cost
    pub fn get_global_cost(&self) -> f32 {
        self.medoids.iter().map(|m| m.get_cost()).sum::<f32>()
    }

    /// return the data id of cluster of rank
    pub fn get_center_id(&self, k : usize) -> Result<DataId, u8> {
        if k < self.medoids.len() {
            Ok(self.medoids[k].get_center_id())
        }
        else {
            Err(1)
        }
    }

    // random initial choice of medoids
    #[allow(unused)]
    fn random_centers_init(&mut self) -> Vec<u32> {
        // we must iterate until we have k different medoids.
        let mut already = vec![false; self.get_size()];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(117);
        let between = Uniform::new::<usize, usize>(0, self.get_size());
        let mut centers = Vec::<u32>::with_capacity(self.nb_cluster);
        // get k different centers
        while centers.len() < self.nb_cluster {
            let j = between.sample(&mut rng);
            if !already[j] {
                already[j] = true;
                centers.push(j as u32);
            }
        }
        centers
    } // end of random_init

    // returns center of each cluster
    fn max_cost_init(&mut self) -> Vec<u32> {
        let mut already = vec![false; self.ids.len()];
        let mut centers = Vec::<u32>::with_capacity(self.nb_cluster);
        //
        // We use a weight rescaling to avoid weight being much larger than distances
        //
        // choose point of maximal weight
        let mut max_item = (0, self.weights[0]);
        for i in 1..self.weights.len() {
            if self.weights[i] > max_item.1 {
                max_item = (i, self.weights[i]);
            }
        }
        already[max_item.0] = true;
        centers.push(max_item.0 as u32);
        log::info!(
            "new center; i : {:6}, weight : {:.3e} dist to previous centers : {:.3e}",
            max_item.0,
            max_item.1,
            0.
        );
        // we maintain costs to current state of centers through iterations.
        let mut costs_to_centers = vec![0.0f32; self.weights.len()];
        for i in 0..self.weights.len() {
            if i == max_item.0 {
                costs_to_centers[i] = 0.;
            } else {
                costs_to_centers[i] = self.distance[[max_item.0, i]] * self.weights[i] as f32;
            }
        }
        // now we create others centers
        loop {
            // search max in costs_to_centers
            max_item = (usize::MAX, 0.0);
            for i in 0..self.weights.len() {
                let cost_i = costs_to_centers[i] as f64;
                if cost_i > max_item.1 {
                    max_item = (i, cost_i as f64);
                }
            }
            assert!(!already[max_item.0]);
            already[max_item.0] = true;
            centers.push(max_item.0 as u32);
            log::info!(
                "new center; i : {:6}, cost : {:.3e} dist to previous centers : {:.3e}",
                max_item.0,
                max_item.1,
                max_item.1 / self.weights[max_item.0]
            );
            //
            if centers.len() == self.nb_cluster {
                break;
            }
            // we need to update costs_to_centers once more
            for i in 0..self.weights.len() {
                if already[i] {
                    costs_to_centers[i] = 0.;
                } else {
                    if self.distance[[max_item.0, i]] * (self.weights[i] as f32)
                        < costs_to_centers[i]
                    {
                        costs_to_centers[i] =
                            self.distance[[max_item.0, i]] * (self.weights[i] as f32);
                    }
                }
            }
        }
        return centers;
    } // end of max_cost_init

    #[allow(unused)]
    fn max_dist_init(&mut self) -> Vec<u32> {
        //
        let mut already = vec![false; self.ids.len()];
        let mut centers = Vec::<u32>::with_capacity(self.nb_cluster);
        //
        // We use a weight rescaling to avoid weight being much larger than distances
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(117);
        let between = Uniform::new::<u32, u32>(0, self.get_size() as u32);
        let first = between.sample(&mut rng);
        // choose point of maximal weight
        let mut max_item = (0, self.weights[0]);
        for i in 1..self.weights.len() {
            if self.weights[i] > max_item.1 {
                max_item = (i, self.weights[i]);
            }
        }
        already[max_item.0] = true;
        centers.push(max_item.0 as u32);
        log::info!(
            "new center; i : {:6}, cost : {:.3e} dist to previous centers : {:.3e}",
            max_item.0,
            0.,
            0.
        );
        // now search a center for each other cluster
        for c in 1..self.nb_cluster {
            let dmax: f32 = 0.;
            // search element furthest away from already chosen centers
            let mut cost_item: (usize, f32, f32) = (usize::MAX, -1., -1.);
            for i in 0..self.distance.ncols() {
                if already[i] {
                    continue;
                }
                for c in 0..centers.len() {
                    let dist = self.distance[[i, centers[c] as usize]];
                    let cost = dist * (self.weights[i] as f32);
                    if cost > cost_item.1 {
                        cost_item = (i, cost, dist);
                    }
                }
            }
            // we furthest point from previous censter, we take it as a new center
            centers.push(cost_item.0 as u32);
            assert!(!already[cost_item.0]);
            already[cost_item.0] = true;
            log::info!(
                "new center; i : {:6}, cost : {:.3e} dist to previous centers : {:.3e}",
                cost_item.0,
                cost_item.1,
                cost_item.2
            );
        }
        //
        return centers;
    }

    // dispatch data to medoids. Returns for each data point cluster number and distance to center of the cluster
    fn dispatch_to_medoids(&mut self, centers: &Vec<u32>) -> MemberDist {
        //
        let membership_dist: Vec<(u32, f32)> = (0..self.get_size())
            .into_par_iter()
            .map(|i| self.find_medoid_for_i(i, centers))
            .collect();
        //
        MemberDist(membership_dist)
    } // end of dispatch_to_medoids


    // find medoid for point i, returns rank of (cluster) center nearest to i
    fn find_medoid_for_i(&self, i: usize, centers: &Vec<u32>) -> (u32, f32) {
        //
        let rowi = self.distance.row(i);
        let mut best_m = 0u32;
        let mut best_dist = rowi[centers[0] as usize];
        for m in 1..self.medoids.len() {
            let test_m = centers[m];
            if rowi[test_m as usize] < best_dist {
                // affect to best medoid index
                best_m = m as u32;
                best_dist = rowi[test_m as usize];
            }
        }
        (best_m, best_dist)
    }

    fn compute_medoids_cost(&self, memberdist: &MemberDist) -> Vec<f32> {
        //
        let mut costs = vec![0.0f32; self.nb_cluster];
        let mut weights = vec![0.0f64; self.nb_cluster];
        let mut cluster_size = vec![0usize; self.nb_cluster];
        //
        for i in 0..memberdist.0.len() {
            let (c, d) = memberdist.0[i];
            costs[c as usize] += (self.weights[i] as f32) * d;
            cluster_size[c as usize] += 1;
            weights[c as usize] += self.weights[i];
        }
        //
        for i in 0..self.nb_cluster {
            log::info!(
                "medoid i : {}, weight : {:.3e} cost : {:.3e} size : {:5} ",
                i,
                weights[i],
                costs[i],
                cluster_size[i]
            );
        }
        //
        costs
    }

    // we mimic kmean update as described in Park-Jun
    // What is the point in cluster that has minimum distance to others indide cluster m.
    // Returns point index (global range 0..self.nb_point) from which points in that cluster has minimum cost to others
    // Cost for center is defined by minimizing cost :  Sum{i in m} w(i) * dist(i,c)
    //
    // This function returns for each cluster a 2-uple containing (center, cluster cost)
    fn from_membership_to_centers(&self, membership: &MemberDist) -> CenterCost {
        //
        // each point i belongs to a cluster, we update the j term contribution inside the same cluster.
        // at end of loop we have contributions of each term to its cluster
        // this function returns cost of cluster having i as member if we consider i as center
        let cost_i = |i| -> f64 {
            let mut cost: f64 = 0.;
            let item: (u32, f32) = membership.0[i];
            let i_cluster: u32 = item.0;
            for j in 0..self.get_size() {
                // if same medoid, update cost
                if j != i && membership.0[j].0 == i_cluster {
                    cost += (self.distance[[i, j]] as f64) * self.weights[j];
                }
            }
            cost
        };
        // TODO: iterate and collect!

        let mut cost: Vec<f32>;
        if self.get_size() <= 1000 {
            cost = (0..self.distance.nrows()).into_iter().map(|_| 0.).collect();
            for i in 0..self.get_size() {
                cost[i] = cost_i(i) as f32;
            }
        } else {
            cost = (0..self.distance.nrows())
                .into_par_iter()
                .map(|i| cost_i(i) as f32)
                .collect();
        }
        //
        let mut centers: Vec<(usize, f32)> = (0..self.nb_cluster)
            .into_iter()
            .map(|_| (usize::MAX, f32::MAX))
            .collect();
        for i in 0..self.get_size() {
            let c = membership.0[i].0;
            assert!((c as usize) < centers.len());
            if cost[i] < centers[c as usize].1 {
                centers[c as usize].1 = cost[i];
                centers[c as usize].0 = i;
            }
        }
        //
        CenterCost(centers)
    } // end of from_membership_to_centers



    // computes statistics (quantiles) of distances to their centers
    // At this final time we can access internal fields membership and medoids state
    // Returns possibly a candidate 2-uple of medoid center to be perturbated.
    fn quality_summary(&self, perturbation_set : &Vec<(usize, usize)>, end : bool) -> Option<(usize, usize)> {
        log::info!("\n\n kmedoids statistics");
        //
        // We compute distance of items to center of their centroid.
        //
        if end {
            let mut q_dist = CKMS::<f32>::new(0.01);
            for i in 0..self.distance.nrows() {
                let m = self.membership[i];
                let c = self.medoids[m as usize].center as usize;
                q_dist.insert(self.distance[[i, c]]);
            }
            println!("\n distance to centroid quantiles at 0.01 :  {:.2e} , 0.025 : {:.2e}, 0.5 : {:.2e}, 0.75 : {:.2e}   0.99 : {:.2e}\n", 
                q_dist.query(0.01).unwrap().1,  q_dist.query(0.025).unwrap().1, 
                q_dist.query(0.5).unwrap().1, q_dist.query(0.75).unwrap().1, q_dist.query(0.95).unwrap().1);
        }
        //
        let mut medoids_size = vec![0u32; self.nb_cluster];
        let mut medoids_dist_mean = vec![0.0f32; self.nb_cluster];
        for i in 0..self.membership.len() {
            let m = self.membership[i] as usize;
            medoids_size[m] += 1;
            medoids_dist_mean[m] += self.distance[[i, self.medoids[m].center as usize]];
        } 
        for m in 0..self.medoids.len() {
            medoids_dist_mean[m] = medoids_dist_mean[m]/medoids_size[m] as f32;
            log::debug!(
                "medoid : {}, size : {:5} , cost {:.2e} , mean dist : {:.2e}",
                m,
                medoids_size[m],
                self.medoids[m].cost,
                medoids_dist_mean[m]
            )
        }
        //
        // if we are at end, we just return
        if end {
            return None;
        }
        //
        // dump info on too near clusters
        //
        let mut couple_opt: Option<(usize, usize)> = None;
        let mut dmin = f32::MAX;
        //
        for i in 0..self.nb_cluster {
            let i_center = self.medoids[i].get_center() as usize;
            for j in 0..i {
                if perturbation_set.last().is_some() &&  *perturbation_set.last().unwrap() == ((j,i)) {
                    continue;
                }
                let d = self.distance[[i_center, self.medoids[j].get_center() as usize]];
                let crit = 2. * d / (medoids_dist_mean[i] + medoids_dist_mean[j]);
                if crit < dmin  {
                    log::debug!(
                        " mixed medoid centers (i,j) : ({}, {}) , crit = {:.2e}",
                        i,
                        j,
                        crit
                    );
                    dmin = crit;
                    couple_opt = Some((i, j));
                }
            }
        }
        //
        return couple_opt;
    } // end of quality_summary


    /// estimate quantiles of distances between data items. We sample if too many couples.
    pub fn quantile_estimator(&self) -> CKMS<f32> {
        log::info!("statistics on weights");
        let mut quantiles = CKMS::<f64>::new(0.01);
        for w in &self.weights {
            quantiles.insert(*w);
        }
        println!("\n weights quantiles at  0.01 :  {:.2e} , 0.025 : {:.2e}, 0.05 : {:.2e}, 0.5 : {:.2e}, 0.75 : {:.2e}   0.99 : {:.2e}\n", 
                quantiles.query(0.01).unwrap().1,  quantiles.query(0.025).unwrap().1,  quantiles.query(0.05).unwrap().1,
                quantiles.query(0.5).unwrap().1, quantiles.query(0.75).unwrap().1, quantiles.query(0.99).unwrap().1);
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2833);
        let nbrow = self.distance.nrows();
        let between = Uniform::new::<usize, usize>(0, nbrow);
        let to_sample = 10000.min(nbrow * (nbrow - 1) / 2);
        log::info!("statistics on distances");
        let mut nb_sampled = 0;
        let mut quantiles = CKMS::<f32>::new(0.01);
        loop {
            let i = between.sample(&mut rng);
            let j = between.sample(&mut rng);
            if i != j {
                quantiles.insert(self.distance[[i, j]]);
                nb_sampled += 1;
                if nb_sampled >= to_sample {
                    break;
                }
            }
        }
        println!("\n distance quantiles at 0.0001 : {:.2e} , 0.001 : {:.2e}, 0.01 :  {:.2e} , 0.025 : {:.2e}, 0.05 : {:.2e},, 0.5 : {:.2e}, 0.75 : {:.2e}   0.99 : {:.2e}\n", 
            quantiles.query(0.0001).unwrap().1, quantiles.query(0.001).unwrap().1,  quantiles.query(0.01).unwrap().1,  quantiles.query(0.025).unwrap().1, 
            quantiles.query(0.05).unwrap().1, quantiles.query(0.5).unwrap().1, quantiles.query(0.75).unwrap().1, quantiles.query(0.99).unwrap().1);
        //
        return quantiles;
    }


    // perturbation of centers of medoids i and j , call dispatch_to_medoids and return new assignment
    // centers i and j are chosen are abnormally close
    fn center_perturbation(
        &mut self,
        (m1, m2): (usize, usize),
        medoids: &mut Vec<Medoid<DataId>>,
    ) -> bool {
        //
        log::info!("in center_perturbation m1 = {}  m2 = {}", m1, m2);
        //
        let mut rng = rand::thread_rng();
        let unif = rand::distributions::Uniform::new(0., 1.);
        //
        //
        let changed: usize;
        if rng.sample(unif) < 0.5 {
            changed = m1;
        } else {
            changed = m2;
        }
        // we try to find in changed the point farthest from center of unchanged!
        let mut max_d = 0.0f32;
        let mut max_i = usize::MAX;
        let old_center = medoids[changed].get_center() as usize;
        for i in 0..self.membership.len() {
            if self.membership[i] as usize == changed {
                let d = self.distance[[i, old_center]];
                if d > max_d {
                    max_d = d;
                    max_i = i;
                }
            }
        }
        //
        let has_changed = medoids[changed].center as usize != max_i;
        log::info!(" center has changed");
        //
        medoids[changed].center = max_i as u32;
        //
        return has_changed;
        //
    } // end of center_perturbation
} // end of impl Kmedoid
