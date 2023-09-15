//! adaptation of Streaming k-means on well clustered data
//! Braverman Meyerson Ostrovski Roytman ACM-SIAM 2011
//! 
//! 


pub struct Bmor<T: Send+Sync> {
    // base number of centers expected
    k : usize,
    //
    nbdata_expected : usize,
    // nb iterations (phases)
    phase : usize,
    // cost multiplicative factor for upper bound of accepted cost at each phase.
    beta : f32,
    // at each phse we have an upper bound for cost.
    phase_cost_upper : f32,
    //  slackness parameters for cost and number of centers accepted
    gamma : f32,
    // current centers, associated to rank in stream (or in data)
    centers : Vec<(usize,Vec<T>)>,
    //
    f_scale : f64,
}  // end of struct Bmor

impl <T: Send+Sync> Bmor<T> {

    /// - k: number of centers
    /// - nbdata : nb data expected,
    /// - gamma 
    pub fn new(k: usize, nbdata : usize, beta : f32) -> Self {
        let gamma = 10.;
        let nb_centers_bound = gamma * (1. + nbdata.ilog2() as f32) * k as f32; 
        let centers = Vec::<(usize,Vec<T>)>::with_capacity(nb_centers_bound.trunc() as usize);
        let phase_cost_upper = 1.;
        let f_scale = phase_cost_upper/ (k as f32 * (1. + nbdata.ilog2() as f32));
        Bmor{k, nbdata_expected : nbdata, phase : 0, beta, phase_cost_upper, gamma, centers, f_scale : f_scale.into()}
    }


    pub fn batch_process(&mut self, data : &Vec<Vec<T>>) {

    }

    pub fn add_data(&mut self, data : &Vec<T>) {

    }


} // end of impl block Bmor
