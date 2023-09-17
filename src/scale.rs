//! scale distance estimation
//! 


use rayon::prelude::*;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use rand::distributions::{Distribution,Uniform};
use quantiles::ckms::CKMS;     // we could use also greenwald_khanna

use hnsw_rs::dist::*;

pub fn scale_estimation<T, Dist : Distance<T>>(nbsample : usize, data : &Vec<Vec<T>>, distance : &Dist) -> CKMS::<f32>
    where   Dist : Sync,
            T: Send+Sync {
                //
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1454691);
    let nbdata = data.len();
    let unif = Uniform::<usize>::new(0, nbdata);
    let couples : Vec<(usize,usize)> = (0..nbsample).into_iter().map(|_| (unif.sample(&mut rng),unif.sample(&mut rng)) ).collect();
    let dvec : Vec<f32> = couples.into_par_iter().map( |(it1,it2)| distance.eval(&data[it1],&data[it2])).collect();
    //
    let mut q_dist = CKMS::<f32>::new(0.01);
    for d in dvec {
        q_dist.insert(d);
    }
    println!("\n distance quantiles at 0.0001 : {:.2e} , 0.001 : {:.2e}, 0.01 :  {:.2e} , 0.99 : {:.2e}   0.999 : {:.2e}\n", 
        q_dist.query(0.0001).unwrap().1, q_dist.query(0.001).unwrap().1,  q_dist.query(0.01).unwrap().1, q_dist.query(0.99).unwrap().1, q_dist.query(0.999).unwrap().1,);

    return q_dist;
}
