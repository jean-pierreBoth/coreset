# coreset

## Introduction 
This crate is devoted to clustering approximation, in metric spaces, of large number data of points.  
Especially we are interested in case where the data cannot be loaded entirely in memory and need a streaming approach.

The method relies on obtaining a coreset for the metric used in the problem.  
A k-coreset is a sampled summary of a much smaller number of points *k*. The points are selected to approximate the cost of dispatching the original dataset to **every** subset of k points.  
Since the selected points have now **weights** attached to the selected points, so we use a weighted point clustering method to produce final clusters.

The crate comes in the form of a library and a specific binary in the subcrate [fromhnsw](#fromhnsw)

## References to implemented algorithms

1. We consider coreset construction as described in the paper:  
    -  New Fraweworks for Offline and Streaming Coreset Constructions.   
           Braverman, Feldman, Lang, Statsman 2022
           [arxiv-v3](https://arxiv.org/abs/1612.00889)



2. The coreset construction relies on  [$\alpha$,$\beta$] approximation in **metric spaces**.  For this step we use the paper :
    - Streaming k-means on well clustered data.  
                Braverman, Meyerson, Ostrovski, Roytman ACM-SIAM 2011 
                [braverman-1](https://web.cs.ucla.edu/~rafail/PUBLIC/116.pdf) or [braverman-2](https://dl.acm.org/doi/10.5555/2133036.2133039)

3. After obtaining the coreset we need an algorithm to provide a k-medoid on weighted data points and check quality of the approximating coreset. We implemented the very simple algorithm (cited in Friedman Hastie Tibshirani **The elements of statistical learning 2001**, Kmedoids paragraph 14.3.10) with the following adaptations:

    - using a greedy initialization like PAM-BUILD
    - takes into accound points weights,
    - random parturbation of cluster pairs with centroids too near of each other.


### Results

Detailed results are given [here](./Results.md).
We run a simple weighted kmean after the coreset construction and compare with those obtained with [par_fastermap](https://docs.rs/kmedoids/0.5.0/kmedoids/fn.par_fasterpam.html) running on the whole data.

#### conclusion
Even with our simplistic weighted kmedoid implementation, the results are, on the average less than 5% above the reference cost obtained by **par_fastermap**, and  within 8% at 2 or 3 std deviations depending on the number of iterations in the kmedoid. 

The number of iterations for the Kmedoid have a small impact on speed and 25 iterations (with 10 clusters asked) are a good compromise.  

**The speed is one or two orders of magnitude faster**.


## Usage 

The data must be associated to a structure implementing the trait **MakeIter**:  

```
pub trait MakeIter {
    /// The identificator of a data
    type Item;
    /// how to get an iterator
    fn makeiter(&self) -> impl Iterator<Item = Self::Item>;
}
```

The algorithm needs more than one pass on the data, so the algorithm takes as argument a structure  providing
an iterator on the data when needed. (Typically the structure could provide file Io to iterates on data, or just a basic Vec of data).  
**An example is found for mnist data** (Cf *module utils::mnistiter*).  

The implementation does the buffering and parallelization internally.
The most synthetic interface is provided in the module *clustercore*, but coreset construction and bmor algorithm can be accessed separately with
corresponding modules.  
The distances are provided by the crate [anndists](https://crates.io/crates/anndists).

## Fromhnsw

The workspace sub-crate *fromhnsw* provides an implementation of the trait *MakeIter* to run the coreset algorithm on data stored in Hnsw structures of the crate [hnsw_rs](https://crates.io/crates/hnsw_rs). A binary *hcore* provides direct coreset or coreset+kmedoid computations with output in the form of a csv file. See the [Readme](./fromhnsw/README.md).

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.