//! scale distance estimation
//!

use rayon::prelude::*;

use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use quantiles::ckms::CKMS;
use rand::distributions::{Distribution, Uniform}; // we could use also greenwald_khanna

use anndists::dist::*;

///  returns quantiles on distances between points.
#[allow(unused)]
pub(crate) fn scale_estimation<T, Dist>(
    nbsample_arg: usize,
    data: &[Vec<T>],
    distance: &Dist,
) -> CKMS<f32>
where
    Dist: Distance<T> + Sync,
    T: Send + Sync,
{
    //
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1454691);
    let nbdata = data.len();
    let unif = Uniform::<usize>::new(0, nbdata);
    //
    let nbsample = nbsample_arg.min(nbdata * nbdata); // useful for tests
    let couples: Vec<(usize, usize)> = (0..nbsample)
        .map(|_| (unif.sample(&mut rng), unif.sample(&mut rng)))
        .filter(|c| c.0 != c.1)
        .collect();
    let dvec: Vec<f32> = couples
        .into_par_iter()
        .map(|(it1, it2)| distance.eval(&data[it1], &data[it2]))
        .collect();
    //
    let mut q_dist = CKMS::<f32>::new(0.01);
    for d in dvec {
        q_dist.insert(d);
    }
    println!("\n distance quantiles at 0.0001 : {:.2e} , 0.001 : {:.2e}, 0.01 :  {:.2e} , 0.5 : {:.2e}, 0.99 : {:.2e}   0.999 : {:.2e}\n", 
        q_dist.query(0.0001).unwrap().1, q_dist.query(0.001).unwrap().1,  q_dist.query(0.01).unwrap().1,  
                    q_dist.query(0.5).unwrap().1, q_dist.query(0.99).unwrap().1, q_dist.query(0.999).unwrap().1);

    q_dist
}

/// sample neighborhood radii.
pub(crate) fn get_neighborhood_size<T, Dist>(
    _nbsample_arg: usize,
    data: &[Vec<T>],
    distance: &Dist,
) -> CKMS<f32>
where
    Dist: Distance<T> + Sync,
    T: Send + Sync,
{
    //
    let nbdata = data.len();
    let unif = Uniform::<usize>::new(0, nbdata);
    // we loop (with sampling) in nb data  and get an idea on neighbours distance for an overall nbdata complexity
    // We use log(nbpoint) as default neighborhood size
    let nb_sample: usize = 1.max(nbdata.ilog2() as usize);
    let explore = |i: usize| -> (f32, f32) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(14547 + i as u64).clone();
        let mut dvec: Vec<f32> = (0..nb_sample)
            .map(|_| {
                loop {
                    let j = unif.sample(&mut rng);
                    if j != i {
                        let dist = distance.eval(&data[i], &data[j]);
                        break dist;
                    }
                }
            })
            .collect();
        dvec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (dvec[0], dvec[1])
    }; // end explore
       //
    let dist_2: Vec<(f32, f32)> = (0..nb_sample).into_par_iter().map(explore).collect();
    //
    let mut q1_dist: CKMS<f32> = CKMS::<f32>::new(0.01);
    let mut q2_dist: CKMS<f32> = CKMS::<f32>::new(0.01);

    for d in dist_2 {
        q1_dist.insert(d.0);
        q2_dist.insert(d.1);
    }
    println!("\n distance quantiles at 0.001 : {:.2e}, 0.01 :  {:.2e} , 0.5 : {:.2e}, 0.99 : {:.2e}   0.999 : {:.2e}\n", 
        q1_dist.query(0.001).unwrap().1,  q1_dist.query(0.01).unwrap().1,  
                    q1_dist.query(0.5).unwrap().1, q1_dist.query(0.99).unwrap().1, q1_dist.query(0.999).unwrap().1);

    println!("\n distance quantiles at 0.001 : {:.2e}, 0.01 :  {:.2e} , 0.5 : {:.2e}, 0.99 : {:.2e}   0.999 : {:.2e}\n", 
    q2_dist.query(0.001).unwrap().1,  q2_dist.query(0.01).unwrap().1,  
                q2_dist.query(0.5).unwrap().1, q2_dist.query(0.99).unwrap().1, q2_dist.query(0.999).unwrap().1);
    //
    q2_dist
} // end of get_neighborhood_size
