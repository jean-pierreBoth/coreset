# coreset

## Introduction 
This crate is devoted clustering approximation of large number data of points.  
Especially we are interested in case where the data cannot be loaded entirely in memory and need a streaming approach.

The method relies on obtaining a coreset for the metric used in the problem. A coreset is a summary of weighted points that can be shown
to cluster as the original data set, but consisiting in mull smaller number of points.  

## References to implemented algorithms

We consider coreset construction as described in the paper:  
 -  New Fraweworks for Offline and Streaming Coreset Constructions.   
           Braverman, Feldman, Lang, Statsman 2022
           [arxiv-v3](https://arxiv.org/abs/1612.00889)



The coreset construction relies on  [$\alpha$,$\beta$] approximation in **metric spaces**.  For this step we use the paper :
 - Streaming k-means on well clustered data.  
                Braverman, Meyerson, Ostrovski, Roytman ACM-SIAM 2011 
                [braverman-1](https://web.cs.ucla.edu/~rafail/PUBLIC/116.pdf) or [braverman-2](https://dl.acm.org/doi/10.5555/2133036.2133039)

After obtaining the coreset we need an algorithm to provide a k-median on weighted data points and check quality of the approximating coreset. We implemented the very simple algorithm: 
 - Park-Jun 
   Simple and Fast algorithm for k-medoids clustering 2009

 We also implemented 
-  Facility Location in sublinear time.   
       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)

The third  highlights the progress made by the 2 first as it requires an order of magnitude more cpu.


## Results on examples

Examples provided are the standard Mnist-digits and Mnist-fashion.

### Streaming k-means on well clustered data

#### running in one pass. (~ 130 facilities)

|  mnist       |  mean entropy  |    cost      |  nb facility | 
|  :---:       |  :---:         |    :---:     |     :---:    |
|   digits     |    0.70        |     2.05     |      170     |
|   fashion    |    0.79        |     1.74     |      129     |

This algorithm runs on both Mnist data in less than a second (0.7) on a i9 laptop.

#### running with post contraction of number of facilities. (~ 70 facilities)

This algorithm runs on both Mnist data in 0.5 second on a i9 laptop.
The mean entropy of labels distributions (10 labels) in each cluster found is between 0.82 for the digits example and 0.84 for the fashion example.

The cost is the mean of L2 distance of each of the  0 images to its nearest facility. The distance is normalized by the number of pixels (coded in the range [0..256])

|  mnist       |  mean entropy  |    cost      |  nb facility | 
|  :---:       |  :---:         |    :---:     |     :---:    |
|   digits     |    0.84        |     2.13     |      75      |
|   fashion    |    0.837       |     1.78     |      75      |
    
Note : the  Badoiu, Czumaj, Indyk, Sohler ICALP 2005 algorithm (at least for our implementation) requires 10s system time and high threading to get similar entropy and costs. 

###  Coreset Construction

The coreset points, with their weights attached are clustered by our simplistic kmedoid algorithm.  
The cost of clustering with these centers is then compared with the cost of clustering the whole original data obtained
with the crate [kmedoids](https://crates.io/crates/kmedoids) using the parallel [par_fastermap](https://docs.rs/kmedoids/0.5.0/kmedoids/fn.par_fasterpam.html).

The computation times, in seconds, given are system time elapsed and total cpu times (to account for parallelism) 


#### Results for coreset construction + basic weighted medoid  (L1 distance) 

The fraction for data subsampling was set 0.11. We asked 10 clusters.

We give the cost of clustering the coreset and the cost of dispatching a posteriori the whole original data to the medoids position obtained via coreset clustering.

|  mnist       | cost (coreset)   | cost (whole data) | time(sys) s        | time(cpu) s |
|  :-------:   |  :----------:    | :---------:       |    :-------------: | :---------: | 
|   digits     |                  |                   |                    |             |
|   fashion    |    2.267 10^6    |    2.277 10^6     |      1             |    12       |




#### Reference results for medoid computations (L1 distance) with par_fastermap

|  mnist       | cost            | time(sys) s        | time(cpu) s |
|  :-------:   |  :----------:   |    :-------------: | :---------: | 
|   digits     |                 |                    |             |
|   fashion    |    2.183 10^6   |      78            |    2212     |





## Usage 

The data must be associated to a structure implementing the trait *IteratorProvider*.  
The algorithm needs to make more than one pass on the data, so the algorithm takes as argument a structure  providing
an iterator on the data when needed. (Typically the structure could provide Io to each data).  
An example is found for mnist data (Cf *module utils::mnistiter*).  

The implementation will do the buffering and parallelization.


## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.