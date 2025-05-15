//! functions used to check coreset1 and kmedoids on mnist files
//!
//!
//!

// to get kmean
use clustering::*;

use cpu_time::ProcessTime;
use std::time::SystemTime;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cmp::Ordering;
use std::iter::Iterator;

use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use ndarray::{Array1, Array2};

use anndists::prelude::*;
use coreset::prelude::*;

use super::iter::*;

use nmi::*;

#[allow(unused)]
pub struct MnistParams {
    algo: Algo,
} // end of MnistParams

impl MnistParams {
    pub fn new(algo: Algo) -> Self {
        MnistParams { algo }
    }
    //
    pub fn get_algo(&self) -> Algo {
        self.algo
    }
}

//================================================================================================

#[allow(unused)]
// computes sum of distance  of coreset points to nearest cluster centers
pub fn dispatch_coreset<Dist>(
    coreset: &CoreSet<usize, f32, Dist>,
    c_centers: &[Vec<f32>],
    distance: &Dist,
    images: &[Vec<f32>],
) -> f64
where
    Dist: Distance<f32> + Send + Sync + Clone,
{
    //
    let mut error: f64 = 0.;
    for (id, w_id) in coreset.get_items() {
        if !w_id.is_finite() {
            log::info!("id : {}, w total : {:?}", id, w_id);
            std::panic!();
        }
        let data = &(images[*id]);
        let (best_c, best_d): (usize, f32) = (0..c_centers.len())
            .map(|i| (i, distance.eval(data, &c_centers[i])))
            .min_by(|(_, d1), (_, d2)| {
                if d1 < d2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap();
        //
        log::info!(
            " core id : {} centroid : {}, dist : {:.3e}, weight : {:.3e} ",
            id,
            best_c,
            best_d,
            w_id
        );
        if !best_d.is_finite() {
            log::info!(
                "coreset point {:?}, \n cluster center : {:?}",
                data,
                c_centers[best_c]
            );
        }
        assert!(best_d.is_finite());
        // TODO: exponent for dist!!!
        error += (w_id * best_d as f64);
    }
    //
    error
}

// computes sum of distance  of all data points cluster centers
// Estimate total error on whole data and assignment for each image
pub fn dispatch_images<Dist>(
    c_centers: &[Vec<f32>],
    distance: &Dist,
    images: &[Vec<f32>],
) -> (f64, Vec<usize>)
where
    Dist: Distance<f32> + Send + Sync + Clone,
{
    //
    log::info!("computing aposteriori cost");
    // return (center rank, distance to center)
    let find_medoid = |data| -> (usize, f64) {
        let (best_c, best_d): (usize, f32) = (0..c_centers.len())
            .map(|i| (i, distance.eval(data, &c_centers[i])))
            .min_by(|(_, d1), (_, d2)| {
                if d1 < d2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap();
        (best_c, best_d as f64)
    };
    //
    //
    let res_vec = (0..images.len())
        .into_par_iter()
        .map(|i| find_medoid(&images[i]))
        .collect::<Vec<(usize, f64)>>();
    //
    let mut cost: f64 = 0.;
    let mut assignment = Vec::<usize>::with_capacity(images.len());
    for (iclust, dist) in res_vec {
        assignment.push(iclust);
        cost += dist;
    }
    //
    println!(" \n ==========================================================");
    println!(" total error distpaching data to centers : {:.3e}", cost);
    //
    (cost, assignment)
}

//

#[allow(unused)]
// call kmedoids to compare
fn kmedoids_reference<Dist>(
    images: &[Vec<f32>],
    labels: &[u8],
    nbcluster: usize,
    distance: &Dist,
) -> (f64, VecAffectation<usize>)
where
    Dist: Distance<f32> + Send + Sync,
{
    //
    log::info!("\n\n entering kmedoids_reference");
    log::info!("==================================");
    //
    let mut rng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(1453731);
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    // compute matrix distance (possibly subsampled)
    let nbpoints = images.len();
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
        row_i
    };
    //
    let rows: Vec<(usize, Array1<f32>)> = (0..nbpoints)
        .into_par_iter()
        .map(|i| (i, compute_row(i)))
        .collect();
    // now we have rows we must transfer into distances
    for (r, v) in &rows {
        assert_eq!(*r, distances_mat.shape()[0]);
        distances_mat.push_row(v.into()).unwrap();
    }
    println!(
        "distance computations  sys time(ms) {:?} cpu time(ms) {:?}",
        sys_now.elapsed().unwrap().as_millis(),
        cpu_start.elapsed().as_millis()
    );
    // choose initialization
    let mut meds = kmedoids::random_initialization(nbpoints, nbcluster, &mut rand::thread_rng());
    //
    let (loss, assignment, _n_iter, _n_swap): (f64, _, _, _) =
        kmedoids::par_fasterpam(&distances_mat, &mut meds, 100, &mut rng);
    println!("\n\n kmedoids reference distance computations + faster pam  sys time(ms) {:?} cpu time(ms) {:?}\n\n ", sys_now.elapsed().unwrap().as_millis(), cpu_start.elapsed().as_millis());
    //
    // compute information merit
    //
    let affectation = VecAffectation::<usize>::new(assignment);
    let reference = VecAffectation::<usize>::new(labels.iter().map(|l| (*l) as usize).collect());
    let contingency =
        Contingency::<VecAffectation<usize>, usize, usize>::new(affectation.clone(), reference);
    let merit = contingency.get_nmi_sqrt();
    //
    println!("faster pam Loss is: {:.3e}", loss);
    println!(
        "faster pam , information merit get_nmi_sqrt version: {:.3e}",
        merit
    );
    println!("=======================================================");
    (loss, affectation)
} // end of kmedoids_reference

//

pub fn coreset1<Dist: Distance<f32> + Sync + Send + Clone>(
    _params: &MnistParams,
    images: &[Vec<f32>],
    labels: &[u8],
    distance: Dist,
) {
    //
    println!("\n\n entering coreset + our kmedoids");
    println!("==================================");
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    // We need to make an iterator producer from data
    let producer = DataForIterator::new(images);
    // allocate a coreset1 structure
    let beta = 2.;
    let gamma = 2.;
    let k = 10; // as we have 10 classes, but this gives a lower bound
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
    let full_dist_name = std::any::type_name::<Dist>();
    let dist_name = full_dist_name.split("::").last().unwrap();
    log::info!("dist name = {:?}", dist_name);
    match dist_name {
        "DistL1" => {
            // going to medoid
            log::info!("\n\n doing kmedoid clustering using L1");
            log::info!("===================================");
            let nb_cluster = 10;
            let mut kmedoids = Kmedoid::new(&coreset, nb_cluster);
            let nbiter = 25;
            log::info!("nb iter = {}", nbiter);
            kmedoids.compute_medians(15);
            let clusters = kmedoids.get_clusters();
            let mut centers = Vec::<Vec<f32>>::with_capacity(nb_cluster);
            for c in clusters {
                let id = c.get_center_id();
                let _label = labels[id];
                let center = images[id].clone();
                centers.push(center);
            }
            println!(
                "coreset + crate::kmedoids  sys time(ms) {:?} cpu time(ms) {:?}",
                sys_now.elapsed().unwrap().as_millis(),
                cpu_start.elapsed().as_millis()
            );
            let (cost, assignment) = dispatch_images(&centers, &distance, images);
            let coreset_affectation = VecAffectation::<usize>::new(assignment);
            let reference =
                VecAffectation::<usize>::new(labels.iter().map(|l| (*l) as usize).collect());
            let contingency = Contingency::<VecAffectation<usize>, usize, usize>::new(
                coreset_affectation.clone(),
                reference,
            );
            let merit = contingency.get_nmi_sqrt();
            log::info!("coreset+kmedoid  data dispatching cost : {:.3e}", cost);
            println!(
                "coreset+kmedoid , information merit get_nmi_sqrt version: {:.3e}",
                merit
            ); // we try to do a direct median clustering with kmedoid crate
            println!("=================================================");
            let (_, kmedoid_affectation) =
                kmedoids_reference(images, labels, nb_cluster, &distance);
            //
            // now we can try to compare the 2 clustering
            //
            let cross_contingency = Contingency::<VecAffectation<usize>, usize, usize>::new(
                coreset_affectation.clone(),
                kmedoid_affectation,
            );
            let merit = cross_contingency.get_nmi_sqrt();
            log::info!("coreset+kmedoid/ faster_pam nmi_sqrt : {:.3e}", merit);
        }

        "DistL2" => {
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
            let (dispatch_error, _assignment) = dispatch_images(&centers, &distance, images);
            log::info!(" coreset dispatching error : {:.3e}", dispatch_error);
            //
            // let dispatch_error = dispatch_coreset(&coreset, &centers, &distance, &images);
            // log::info!(" coreset dispatching error : {:.3e}", dispatch_error);
        }
        _ => {
            log::info!("no postprocessing for distance {:?}", dist_name);
        }
    }
} // end of coreset1
