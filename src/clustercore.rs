//! This module provides a simple user interface for clustering data via a coreset.  
//! The data must be accessed via an iterator see [iterprovider]
//! It chains the bmor and algorithms and  ends with a pass dispatching all data
//! to their nearest cluster deduced from the coreset clustering, recomputing global
//! cost and storing membership
//! 

#![allow(unused)]   // temporary


use std::collections::HashMap;
use std::hash::Hash;
use dashmap::DashMap;

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use hnsw_rs::dist::*;


use crate::sensitivity::*;
// use crate::facility::*;
use crate::iterprovider::*;
use crate::wkmedian::*;

#[derive(Copy,Clone)]
pub struct BmorArg {
    nb_data_expected : usize,
    /// drives upper cost acceptable through iterations. 2. is a standard default.
    beta : f64,
    /// 
    gamma : f64
}


impl Default for BmorArg {

    fn default() -> Self {
        BmorArg{nb_data_expected : 1_000_000, beta: 2. , gamma: 2.}
    }
}

//==================================================================

pub struct ClusterCoreset<DataId :  std::fmt::Debug + Eq+ std::hash::Hash + Clone + Send + Sync> {
    ///
    nb_cluster : usize,
    /// fraction of data to keep in coreset
    fraction : f64,
    ///
    bmor_arg: BmorArg,
    /// To store kmedoid result
    kmedoids : Option<Kmedoid<DataId>>, 
    /// associate each DataId to its medoid rank (computed by function )
    ids_to_cluster : Option<HashMap<DataId, usize >>,
    //
}

impl <DataId> ClusterCoreset<DataId>                    
    where DataId : std::fmt::Debug + Default + Eq + Hash + Send + Sync + Clone
{

    pub fn new(nb_cluster : usize, fraction : f64, bmor_arg : BmorArg) -> Self {
        ClusterCoreset{nb_cluster, fraction, bmor_arg, kmedoids : None, ids_to_cluster : None}
    }

    pub fn compute<T, Dist, IterProducer>(&self, distance : Dist, iter_producer : IterProducer) 
            where T : Send + Sync + Clone,
                Dist : Distance<T> + Send + Sync + Clone,
                IterProducer : IterProvider<DataId = DataId, DataType = Vec<T>>  {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let mut coreset1 = Coreset1::<DataId, T, Dist>::new(self.nb_cluster, 
                                                                self.bmor_arg.nb_data_expected, 
                                                                self.bmor_arg.beta, self.bmor_arg.gamma,
                                                                distance.clone());
        //
        let result = coreset1.make_coreset(&iter_producer, self.fraction);
        if result.is_err() {
            log::error!("construction of coreset1 failed");
        }
        let coreset = result.unwrap();
        log::info!("coreset1 nb different points : {}", coreset.get_nb_points());
        //
        log::info!("\n\n doing kmedoid clustering using distance : {}", std::any::type_name::<Dist>());
        log::info!("===================================");
        let nb_cluster = self.nb_cluster;
        let mut kmedoids = Kmedoid::new(&coreset, nb_cluster);
        let (nb_iter , cost) = kmedoids.compute_medians();
        // TODO: we have coreset and kmedoids we must store center (Vec<T>) of each medoid!

        std::panic!("not yet");
        let cpu_time: Duration = cpu_start.elapsed();
        log::info!(" kmedoids finished at nb_iter : {}, cost = {:.3e}", nb_iter, cost);
        log::info!(" sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_time.as_millis()); 
        //
        log::info!(" dispatching all data to their cluster, needs one pass more through data");
        let cpu_time: Duration = cpu_start.elapsed();
        log::info!("\n  ClusterCoreset::compute sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_time.as_millis()); 
    } // end of compute



    fn dispatch<T, Dist, IterProducer>(&mut self, kmedoid : &Kmedoid<DataId>, iter_producer : &IterProducer) 
                where T : Send + Sync + Clone,
                Dist : Distance<T> + Send + Sync + Clone,
                IterProducer : IterProvider<DataId = DataId, DataType = Vec<T>> {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        // We must bufferise and make distance computation and use a concurrent HashMap (We did it in Bmor)
        //
        let mut data_iter = iter_producer.makeiter();
        let buffer_size = 50000;   //  TODO: ?
        let mut map = DashMap::<DataId, usize>::with_capacity(self.bmor_arg.nb_data_expected); //TODO:  at this step we know nb_data!
        //
        let dispatch_i = | (id, data) : (DataId, Vec<T>) | -> usize  {
            // get neaarest medoid. We can retrieve Vec<T> for each medoid from coreset , so we must get access to it
       
            std::panic!("not yet");
        };
        //
        loop {
            let buffres = self.get_buffer_data::<T, Dist>(buffer_size, &mut data_iter);
            if buffres.is_err() {
                break;
            }
            let (ids, datas) = buffres.unwrap();
            // now a // call to closure
        }

        log::info!(" end of data dispatching dispatching all data to their cluster");
        let cpu_time: Duration = cpu_start.elapsed();
        log::info!("\n  ClusterCoreset::dispatch sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_time.as_millis()); 
    } // end of dispatch


    /// use iterator to return a block of data
    fn get_buffer_data<T, Dist>(&self, buffer_size : usize, data_iter : &mut impl Iterator<Item=(DataId, Vec<T>)>) -> Result<(Vec<DataId>, Vec<Vec<T>>), u32>
        where      T : Send + Sync + Clone,
                Dist : Distance<T> + Send + Sync + Clone {
        //
        let mut datas = Vec::<Vec<T>>::with_capacity(buffer_size);
        let mut ids = Vec::<DataId>::with_capacity(buffer_size);
        //
        loop {
            let data_opt = data_iter.next();
            match data_opt {
                Some((id, data)) => {
                    // insert
                    datas.push(data);
                    ids.push(id);
                    if datas.len() == buffer_size {
                        break;
                    }
                }
                _ => {
                    break;
                }
            } // end ma
        } // end loop
        // 
        assert_eq!(ids.len(), datas.len());
        if ids.len() > 0 {
            return Ok((ids, datas));
        }
        else {
            return Err(0);
        }
    } // end of get_buffer_data



} // end of impl ClusterCorese
