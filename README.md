# coreset

## Introduction 
This crate is devoted to clustering approximation of large number data of points.  
Especially we are interested in case where the data cannot be loaded entirely in memory and need a streaming approach.

The method relies on obtaining a coreset for the metric used in the problem. A coreset is a summary of a much smaller number of points.
The points have a weight attached and are selected to approximate the cost of dispatching the original dataset to every subset of k points.
It is thus possible to get an approximate clustering.

## References to implemented algorithms

1. We consider coreset construction as described in the paper:  
    -  New Fraweworks for Offline and Streaming Coreset Constructions.   
           Braverman, Feldman, Lang, Statsman 2022
           [arxiv-v3](https://arxiv.org/abs/1612.00889)



2. The coreset construction relies on  [$\alpha$,$\beta$] approximation in **metric spaces**.  For this step we use the paper :
    - Streaming k-means on well clustered data.  
                Braverman, Meyerson, Ostrovski, Roytman ACM-SIAM 2011 
                [braverman-1](https://web.cs.ucla.edu/~rafail/PUBLIC/116.pdf) or [braverman-2](https://dl.acm.org/doi/10.5555/2133036.2133039)

3. After obtaining the coreset we need an algorithm to provide a k-median on weighted data points and check quality of the approximating coreset. We implemented the very simple algorithm: 
    - Friedman Hastie Tibshirani The elements of statistical learning 2001

    using a greedy initialization like PAM-BUILD and first try to randomize cluster pairs too near of each other.

4. We also implemented 
   -  Facility Location in sublinear time.   
       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)

The third  highlights the progress made by the 2 first as it requires an order of magnitude more cpu.


## Results

Examples provided are the standard Mnist-digits and Mnist-fashion.

###  Coreset Construction

The coreset points, with their weights attached are clustered by our simplistic kmedoid algorithm.  
The cost of clustering with these centers is then compared with the cost of clustering the whole original data obtained
with the crate [kmedoids](https://crates.io/crates/kmedoids) using the parallel [par_fastermap](https://docs.rs/kmedoids/0.5.0/kmedoids/fn.par_fasterpam.html).

The computation times, in seconds, given are system time elapsed and total cpu times (to account for parallelism) 


#### Results for coreset construction + basic weighted medoid  (L1 distance) 

The fraction for data subsampling was set 0.11. We asked 10 clusters.


We give the me cost of clustering the coreset and the cost of dispatching a posteriori the whole original data to the medoids position obtained via coreset clustering.  
As the results are random we give  the results in the form (mean +-standard deviation) obtained on a sample of 20 computations

|  mnist       | cost (coreset)         | cost (whole data)     | time(sys) s   | time(cpu) s |
|  :-------:   |  :--------------:      | :-------------:       |  :---------:  | :---------: | 
|   digits     | (1.877 +- 0.03) 10^6   | (1.883 +- 0.031) 10^6 |      1        |    12.4     |
|   fashion    | (2.272 +- 0.0516) 10^6 | (2.277+- 0.045) 10^6  |      1        |    12       |



#### Reference results for medoid computations (L1 distance) with par_fastermap

The timings takes into account the computing of the distance matrix (fully multithreaded)
|  mnist       | cost            | time(sys) s        | time(cpu) s |
|  :-------:   |  :----------:   |    :-------------: | :---------: | 
|   digits     |    1.789 10^6   |      55            |    1660     |
|   fashion    |    2.183 10^6   |      78            |    2212     |

#### Conclusion:

**The results are, on the average at 5% above the reference cost obtained by faster map, and consistently under 10% even with our simplistic weighted kmedoid implementation.  
The speed is one or two orders of magnitude faster**.


##### Results on the [$\alpha$,$\beta$] approximation can be found [here](./bmor.md)




## Usage 

The data must be associated to a structure implementing the trait *IteratorProvider*.  
The algorithm needs to make more than one pass on the data, so the algorithm takes as argument a structure  providing
an iterator on the data when needed. (Typically the structure could provide file Io to each data).  
An example is found for mnist data (Cf *module utils::mnistiter*).  

The implementation will do the buffering and parallelization.


## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.