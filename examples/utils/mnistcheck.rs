//! functions used to check coreset1 and kmedoids on mnist files
//! 
//! 
//! 

// to get kmean
use clustering::*;


use std::time::SystemTime;
use cpu_time::ProcessTime;

use rayon::iter::{ParallelIterator, IntoParallelIterator};

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use ndarray::{Array1,Array2};

use std::iter::Iterator;
use hnsw_rs::prelude::*;
use coreset::prelude::*;

use super::mnistiter::*;


use std::cmp::Ordering;

#[allow(unused)]
pub struct MnistParams {
    algo : Algo
} // end of MnistParams

impl MnistParams {
    pub fn new(algo : Algo) -> Self {
        MnistParams{algo}
    }
    //
    pub fn get_algo(&self) -> Algo { self.algo}
}

//================================================================================================

#[allow(unused)]
// computes sum of distance  of coreset points to nearest cluster centers
pub fn dispatch_coreset<Dist>(coreset : &CoreSet<f32, Dist>,  c_centers : &Vec<Vec<f32>>, distance : &Dist, images : &Vec<Vec<f32>>) -> f64 
    where Dist : Distance<f32> + Send + Sync + Clone {
    //
    let mut error : f64 = 0.;
    for (id, w_id) in coreset.get_items() {
        if !w_id.is_finite() {
            log::info!("id : {}, w total : {:?}", id, w_id);
            std::panic!();
        }
        let data = &(images[*id]);
        let (best_c, best_d) : (usize, f32) = (0..c_centers.len()).into_iter()
            .map(|i| (i, distance.eval(data, &c_centers[i])))
            .min_by(| (_,d1), (_,d2)| if d1 < d2 
                    {Ordering::Less} 
                else 
                    {Ordering::Greater })
            .unwrap();
        //
        log::info!(" core id : {} centroid : {}, dist : {:.3e}, weight : {:.3e} ", id, best_c, best_d, w_id);
        if !best_d.is_finite() {
            log::info!("coreset point {:?}, \n cluster center : {:?}", data , c_centers[best_c]);
        }
        assert!(best_d.is_finite());
        // TODO: exponent for dist!!!
        error += (w_id * best_d as f64) as f64;
    }
    //
    error
}



// computes sum of distance  of all data points cluster centers
// Estimate total error on whole data
pub fn dispatch_images<Dist>(c_centers : &Vec<Vec<f32>>, distance : &Dist, images : &Vec<Vec<f32>>) -> f64 
    where Dist : Distance<f32> + Send + Sync + Clone {
    //
    log::info!("computing aposteriori cost");
    //
    let find_medoid = | data| -> (usize, f64) {
        let (best_c, best_d) : (usize, f32) = (0..c_centers.len()).into_iter()
        .map(|i| (i, distance.eval(data, &c_centers[i])))
        .min_by(| (_,d1), (_,d2)| if d1 < d2 
                {Ordering::Less} 
            else 
                {Ordering::Greater })
        .unwrap();
        (best_c, best_d as f64)
    };
    //
    //
    let cost = (0..images.len()).into_par_iter()
        .map(|i| find_medoid(&images[i]).1)
        .sum::<f64>();
    //
    println!(" \n ==========================================================");
    println!(" total error distpaching data to centers : {:.3e}", cost);
    println!(" ==========================================================");
    //
    cost
}

#[allow(unused)]
// call kmedoids to compare
fn kmedoids_reference<Dist>(images : &Vec<Vec<f32>>, _labels : &Vec<u8>, nbcluster : usize, distance : &Dist) 
            where Dist : Distance<f32> + Send + Sync     {
    //
    log::info!("\n\n entering kmedoids_reference");
    log::info!("==================================");
    //
    let mut rng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(1453731);
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    // compute matrix distance (possibly subsampled)
    let nbpoints =  images.len();
    // allocates to zero rows. We will computes rows in //
    let mut distances_mat = Array2::<f32>::zeros((0, nbpoints));
    //
    let compute_row = |i| -> Array1<f32> {
        let mut row_i = Array1::zeros(nbpoints);
        for j in 0..nbpoints {
            if j != i {
                row_i[j] = distance.eval(&images[i], &images[j]);
            }
        }
        return row_i;
    };
    //
    let rows : Vec<(usize, Array1<f32>)>= (0..nbpoints).into_par_iter().map(|i| (i, compute_row(i))).collect();
    // now we have rows we must transfer into distances
    for (r,v) in &rows {
        assert_eq!(*r,distances_mat.shape()[0]);
        distances_mat.push_row(v.into()).unwrap();
    }
    println!("distance computations  sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_start.elapsed().as_millis());
    // choose initialization
    let mut meds = kmedoids::random_initialization(nbpoints, nbcluster, &mut rand::thread_rng());
    //
    let (loss, _assi, _n_iter, _n_swap): (f64, _, _, _) = kmedoids::par_fasterpam(&distances_mat, &mut meds, 100, &mut rng);
    println!("faster pam Loss is: {}", loss);
    println!("\n\n kmedoids reference distance computations + faster pam  sys time(ms) {:?} cpu time(ms) {:?}\n\n ", sys_now.elapsed().unwrap().as_millis(), cpu_start.elapsed().as_millis());
} // end of kmedoids_reference



pub fn coreset1<Dist : Distance<f32> + Sync + Send + Clone>(_params :&MnistParams, images : &Vec<Vec<f32>>, _labels : &Vec<u8>, distance : Dist) {
    //
    println!("\n\n entering coreset + our kmedoids");
    println!("==================================");
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    // We need to make an iterator producer from data
    let producer = IteratorProducer::new(images);
    // allocate a coreset1 structure
    let beta = 2.;
    let gamma = 2.;
    let k = 10;  // as we have 10 classes, but this gives a lower bound
    let mut core1 = Coreset1::new(k, images.len(), beta, gamma, distance.clone());
    //
    let res = core1.make_coreset(&producer, 0.11);
    if res.is_err() {
        log::error!("construction of coreset1 failed");
    }
    let coreset = res.unwrap();
    // get some info
    log::info!("coreset1 nb different points : {}", coreset.get_nb_points());
    // 
    let dist_name = std::any::type_name::<Dist>();
    log::info!("dist name = {:?}", dist_name);
    match dist_name {
        "hnsw_rs::dist::DistL1" => {
            // going to medoid
            log::info!("\n\n doing kmedoid clustering using L1");
            log::info!("===================================");
            let nb_cluster = 10;
            let mut kmedoids = Kmedoid::new(&coreset, nb_cluster);
            kmedoids.compute_medians();
            let clusters = kmedoids.get_clusters();
            let mut centers = Vec::<Vec<f32>>::with_capacity(nb_cluster);
            for c in clusters {
                let id = c.get_center_id();
                let _label = _labels[id];
                let center = images[id].clone();
                centers.push(center);
            }
            println!("coreset + crate::kmedoids  sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_start.elapsed().as_millis());
            let dispatch_error = dispatch_images(&centers, &distance, &images);
            log::info!(" original data dispatching error : {:.3e}", dispatch_error);
            // we try to do a direct median clustering with kmedoid crate
//            kmedoids_reference(images, _labels, nb_cluster, &distance);
        }

        "hnsw_rs::dist::DistL2" => {
            // going to kmean
            log::info!("doing kmean clustering on whole data .... takes time");
            let nb_iter = 50;
            let nb_cluster = 10;
            let clustering = kmeans(nb_cluster, images, nb_iter);
            // compute error
            let centroids = &clustering.centroids;
            // conver centroids to vectors
            let mut centers = Vec::<Vec<f32>>::with_capacity(nb_cluster);
            for c in centroids {
                let dim = c.dimensions();
                let mut center = Vec::<f32>::with_capacity(dim);
                for i in 0..dim {
                    center.push(c.at(i) as f32);
                }
                centers.push(center);
            }
            let elements = clustering.elements;
            let membership = clustering.membership;
            let mut error = 0.0;
            for i in 0..elements.len() {
                let cluster = membership[i];
                error += distance.eval(&elements[i], &centers[cluster]);
            }
            log::info!("kmean error : {:.3e}", error / images.len() as f32);
            // now we must dispatch our coreset to centers and see what error we have...
            let dispatch_error = dispatch_images(&centers, &distance, &images);
            log::info!(" coreset dispatching error : {:.3e}", dispatch_error);
            //
            // let dispatch_error = dispatch_coreset(&coreset, &centers, &distance, &images);
            // log::info!(" coreset dispatching error : {:.3e}", dispatch_error);
        }
        _ => { log::info!("no postprocessing for distance {:?}", dist_name); }
    }
} // end of coreset1

