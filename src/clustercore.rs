//! This module provides a simple user interface for clustering data via a coreset.  
//! The data must be accessed via an iterator, see [makeiter](super::makeiter).  
//! It chains the bmor,coreset. Then the coreset is clustered with kmedoid algorithms and ends with a pass dispatching all data
//! to their nearest coreset clustering center, recomputing global
//! cost and storing membership
//!
//! Clusters are dumped in a csv file name "clustercoreset.csv".  
//! Each line consists in the DataId of an item, the DataId of its cluster center
//!

use std::collections::HashMap;
use std::hash::Hash;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use cpu_time::ProcessTime;
use num_cpus;
use std::time::SystemTime;

use std::io::Write;

use anndists::dist::*;

use crate::sensitivity::*;
// use crate::facility::*;
use crate::makeiter::*;
use crate::wkmedian::*;

/// Bmor parameters driving the coreset construction
/// See [Bmor](super::bmor::Bmor)
#[derive(Copy, Clone)]
pub struct BmorArg {
    nb_data_expected: usize,
    /// drives upper cost acceptable through iterations. 2. is a standard default.
    beta: f64,
    //
    gamma: f64,
}

impl BmorArg {
    /// nb_data_expected : number of data expected
    pub fn new(nb_data_expected: usize, beta: f64, gamma: f64) -> Self {
        BmorArg {
            nb_data_expected,
            beta,
            gamma,
        }
    }
}

impl Default for BmorArg {
    fn default() -> Self {
        BmorArg {
            nb_data_expected: 1_000_000,
            beta: 2.,
            gamma: 2.,
        }
    }
}

//==================================================================

pub struct ClusterCoreset<DataId: std::fmt::Debug + Eq + std::hash::Hash + Clone + Send + Sync, T> {
    //
    nb_cluster: usize,
    /// fraction of data to keep in coreset
    fraction: f64,
    //
    bmor_arg: BmorArg,
    // total nb data processed
    nb_data: usize,
    /// To store kmedoid result
    kmedoids: Option<Kmedoid<DataId, T>>,
    /// associate each DataId to its medoid rank (computed by function dispatch)
    ids_to_cluster: Option<HashMap<DataId, DataId>>,
}

impl<DataId, T> ClusterCoreset<DataId, T>
where
    DataId: Default + Eq + Hash + Send + Sync + Clone + std::fmt::Debug,
    T: Clone + Send + Sync + std::fmt::Debug,
{
    /// - nb_cluster
    /// - fraction : fraction of data to keep in coreset.  
    ///   The fraction is chosen so that
    ///   the subsequent k-medoids phase is computationally tractable and yet sufficient to
    ///   represent the whole data. (Typically for Mnist data ~0.1 is a valid choice)
    /// - bmor_arg : defines parameter to Bmor pass
    pub fn new(nb_cluster: usize, fraction: f64, bmor_arg: BmorArg) -> Self {
        ClusterCoreset {
            nb_cluster,
            fraction,
            bmor_arg,
            nb_data: 0,
            kmedoids: None,
            ids_to_cluster: None,
        }
    }

    /// computes coreset and kmedoid clustering.  
    /// - distance : the metric to use
    /// - nb_iter : the maximal number of iterations in kmedoid.  
    ///    
    /// **Note:**  
    /// Just the DataId of the center are stored in kmedoid structure, not the Data vector to spare memory.  
    /// To extract the data vectors, call [dispatch](Self::dispatch()) which retrive the data vectors and computes the clustering cost with respect to the whole data.
    /// It needs one more pass on the data.
    pub fn compute<Dist, IterProducer>(
        &mut self,
        distance: Dist,
        nb_iter: usize,
        iter_producer: &IterProducer,
    ) where
        Dist: Distance<T> + Send + Sync + Clone,
        IterProducer: MakeIter<Item = (DataId, Vec<T>)>,
    {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let mut coreset1 = Coreset1::<DataId, T, Dist>::new(
            self.nb_cluster,
            self.bmor_arg.nb_data_expected,
            self.bmor_arg.beta,
            self.bmor_arg.gamma,
            distance.clone(),
        );
        //
        let result = coreset1.make_coreset(iter_producer, self.fraction);
        log::info!(
            "make_coreset done sys time {}, cpu time {}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        if result.is_err() {
            log::error!("construction of coreset1 failed");
        }
        let coreset = result.unwrap();
        log::info!("coreset1 nb different points : {}", coreset.get_nb_points());
        //
        log::info!(
            "\n\n doing kmedoid clustering using distance : {}",
            std::any::type_name::<Dist>()
        );
        let nb_cluster = self.nb_cluster;
        let mut kmedoids = Kmedoid::new(&coreset, nb_cluster);
        let (nb_iter, cost) = kmedoids.compute_medians(nb_iter);
        // TODO: we have coreset and kmedoids we must store center (Vec<T>) of each medoid!
        self.nb_data = coreset1.get_nb_data();
        //
        log::info!(
            " kmedoids finished at nb_iter : {}, cost = {:.3e}",
            nb_iter,
            cost
        );
        log::info!(
            " ClusterCoreset::compute (coreset+kmedoids sys time(ms) {:?} cpu time(ms) {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        //
        self.kmedoids = Some(kmedoids);
    } // end of compute

    //

    /// Once you have Kmedoid, you can compute the clustering cost for the whole data, not just the coreset.
    /// This function can also fill in  [Kmedoid] structure the data vector associated to each center, see [Kmedoid::get_cluster_center]
    pub fn dispatch<Dist, IterProducer>(&mut self, distance: &Dist, iter_producer: &IterProducer)
    where
        T: Send + Sync + Clone,
        Dist: Distance<T> + Send + Sync + Clone,
        IterProducer: MakeIter<Item = (DataId, Vec<T>)>,
    {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        // We must bufferize and make distance computation
        //
        let mut data_iter = iter_producer.makeiter();
        let nb_cpus = num_cpus::get();
        let buffer_size = 5000 * nb_cpus;
        let mut map_to_medoid = HashMap::<DataId, DataId>::with_capacity(self.nb_data);
        // We must retrive datas corresponding to medoid centers
        self.get_kmedoids().retrieve_cluster_centers(iter_producer);
        let centers = self.kmedoids.as_ref().unwrap().get_centers().unwrap();
        if centers.is_empty() {
            log::error!("ClusterCore::dispatch, kmedoids centers have not yet been computed");
            std::process::exit(1);
        }
        //
        // This function returns for each data (id,data) a triplet (id, rank of nearest center found and distance to its cluster center)
        //
        let dispatch_i = |(id, data): (DataId, &Vec<T>)| -> (DataId, usize, f32) {
            // get nearest medoid. We can retrieve Vec<T> for each medoid from coreset , so we must get access to it
            assert!(!data.is_empty());
            let dists: Vec<f32> = centers.iter().map(|c| distance.eval(data, c)).collect();
            let mut dmin = f32::MAX;
            let mut imin = usize::MAX;
            for (i, d) in dists.iter().enumerate() {
                if *d < dmin {
                    dmin = *d;
                    imin = i;
                }
            }
            if imin >= dists.len() {
                log::error!("\n dispatch failed for id {:?}, FATAL EXITING", id);
                std::process::exit(1);
            }
            //
            (id, imin, dmin)
        };
        let mut dispatching_cost: f64 = 0.;
        let mut nb_total_data = 0usize;
        //
        loop {
            let buffres = self.get_buffer_data(buffer_size, &mut data_iter);
            if buffres.is_err() {
                break;
            }
            let ids_datas = buffres.unwrap();
            nb_total_data += ids_datas.len();
            // dispatch buffer
            let res_dispatch: Vec<(DataId, usize, f32)> = ids_datas
                .into_par_iter()
                .map(|(i, d)| dispatch_i((i, &d)))
                .collect();
            for (id, cluster_rank, d) in res_dispatch {
                let c_id_res = self.kmedoids.as_ref().unwrap().get_center_id(cluster_rank);
                if c_id_res.is_err() {
                    log::error!("cannot get center of cluster n° : {}", cluster_rank);
                    std::process::exit(1);
                }
                let c_id = c_id_res.unwrap();
                map_to_medoid.insert(id, c_id);
                dispatching_cost += d as f64;
            }
        }
        println!(
            "\n end of data dispatching dispatching all data to their cluster, global cost : {:.3e}, cost by data : {:.3e}",
            dispatching_cost,
            dispatching_cost/ nb_total_data as f64
        );
        //
        // dump clusters DataId info
        //
        self.ids_to_cluster = Some(map_to_medoid);
        let _ = self.dump_clusters();
        //
        log::info!(
            "\n  ClusterCoreset::dispatch sys time(ms) {:?} cpu time(ms) {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
    } // end of dispatch

    //

    /// Dumps in file for each dataId, the DataId of corresponding cluster center. If Ok returns number of record dumped.  
    /// The dump is a csv file whose name is *clustercoreset-pid.csv* where pid is the pid of the process.
    /// Each line consists in 2 DataId, the first one identifies a data point and the second the DataId of the center of its corresponding cluster.  
    /// This function requires [dispatch][Self::dispatch()] to have been called previously
    pub fn dump_clusters(&self) -> anyhow::Result<usize> {
        //
        let mut name = String::from("clustercoreset");
        name.push_str(".csv");
        let file = std::fs::File::create(&name)?;
        let mut bufw = std::io::BufWriter::new(file);
        let mut nb_record = 0;
        //
        if self.ids_to_cluster.is_none() {
            log::error!(
                "ClusterCorest::dump_clusters: The method dispatch should have been alled before"
            );
            return Err(anyhow::anyhow!(
                "ClusterCorest::dump_clusters: The method dispatch should have been alled before"
            ));
        }
        for (d, c) in self.ids_to_cluster.as_ref().unwrap() {
            writeln!(bufw, "{:?},{:?}\n", d, c).unwrap();
            nb_record += 1;
        }
        bufw.flush().unwrap();
        log::info!(
            "clustercorest, dumping cluster info in file {:?} , nb_record : {:?} ",
            name,
            nb_record
        );
        //
        Ok(nb_record)
    }

    /// use iterator to return a block of data
    fn get_buffer_data(
        &self,
        buffer_size: usize,
        data_iter: &mut impl Iterator<Item = (DataId, Vec<T>)>,
    ) -> Result<Vec<(DataId, Vec<T>)>, u32> {
        //
        let mut ids_datas = Vec::<(DataId, Vec<T>)>::with_capacity(buffer_size);
        //
        loop {
            let data_opt = data_iter.next();
            match data_opt {
                Some((id, data)) => {
                    // insert
                    ids_datas.push((id, data));
                    if ids_datas.len() == buffer_size {
                        break;
                    }
                }
                _ => {
                    break;
                }
            } // end ma
        } // end loop
          //
        if !ids_datas.is_empty() {
            Ok(ids_datas)
        } else {
            Err(0)
        }
    } // end of get_buffer_data

    fn get_kmedoids(&mut self) -> &mut Kmedoid<DataId, T> {
        self.kmedoids.as_mut().unwrap()
    }
} // end of impl ClusterCorese
