//! sampling from a discrete probability using inverse repartition function method
//! 

use num_traits::float::Float;
use rand::Rng;
use rand::distributions::Uniform;

use std::cmp::Ordering;

pub struct DiscreteProba<F:Float + rand_distr::uniform::SampleUniform> {
    repartition : Vec<F>,
    unif : Uniform<F>,
}

impl <F:Float + std::fmt::Debug + rand_distr::uniform::SampleUniform> DiscreteProba<F> {

    pub fn new(probas : &Vec<F>) -> Self {
        let size = probas.len() + 1;
        let mut repartition = Vec::<F>::with_capacity(size);
        let mut cumul = F::zero();
        //
        repartition.push(F::zero());
        for v in probas {
            assert!( *v >= F::zero());
            cumul = cumul + *v;
            repartition.push(cumul);
        }
        // cumulate
        for i in 1..repartition.len() {
            repartition[i] = repartition[i]/cumul;
            assert!(repartition[i] > repartition[i-1]);
        }
        let last = repartition.len()-1;
        repartition[last] = F::one();
        //
        DiscreteProba{repartition, unif : Uniform::<F>::new(F::zero(), F::one())}
    }

    /// returns slot sampled and associated proba
    pub fn sample<R:Rng>(&self, rng : &mut R) -> (usize , F) {
        //
        let xsi : F = rng.sample(&self.unif);
        log::trace!("sampled xsi : {:?}", xsi);
        let slot = self.repartition.binary_search_by(|w| {
            if *w <= xsi {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
        .unwrap_err();
        // 
        assert!(slot >= 1);
        //
        return (slot, self.get_proba(slot));
    } // end of sample



    /// get probability of a slot
    pub fn get_proba(&self, slot : usize) -> F {
        assert!(slot < self.repartition.len());
        self.repartition[slot] -  self.repartition[slot-1]
    }

    /// returns repartition function
    pub fn get_repartition_function(&self) -> &Vec<F> {
        &self.repartition
    }


} // end of impl block for DiscreteProba



#[cfg(test)]

mod tests {


    use super::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::SeedableRng;


    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test1() {
        log_init_test();
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(14537);

        let p1: Vec<f32> = vec![0.1, 0.2, 0.2, 0.1, 0.3, 0.05, 0.05];
        //
        let proba = DiscreteProba::new(&p1);
        log::debug!("repartiton function : {:?}", proba.get_repartition_function());
        //
        for _ in 0..10 {
            let (slot, weight) = proba.sample(&mut rng);
            log::debug!(" slot : {}, weight : {:?}", slot, weight);
        }
    } // end of test1


} // end of mod tests