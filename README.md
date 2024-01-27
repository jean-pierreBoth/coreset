# coreset

Some exploration of median [$\alpha$,$\beta$] approximation in **metric spaces** and coreset construction based on the following papers:



1. Streaming k-means on well clustered data.  
        Braverman, Meyerson, Ostrovski, Roytman ACM-SIAM 2011 
        [braverman-1](https://web.cs.ucla.edu/~rafail/PUBLIC/116.pdf) or
        [braverman-2](https://dl.acm.org/doi/10.5555/2133036.2133039)

2. New Fraweworks for Offline and Streaming Coreset Constructions.   
        Braverman, Feldman, Lang, Statsman 2022
        [arxiv-v3](https://arxiv.org/abs/1612.00889)


3. Facility Location in sublinear time.   
       Badoiu, Czumaj, Indyk, Sohler ICALP 2005
       [indyk](https://people.csail.mit.edu/indyk/fl.pdf)

The first algorithm is used a [$\alpha$,$\beta$] approximation used in the coreset construction described in the second reference.
The third  highlights the progress made by the 2 first as it requires an order of magnitude more cpu.

A weighted k-median is also provided to run on weighted points produced in the coreset construction.

## Examples

Examples provided are the standard Mnist-digits and Mnist-fashion.

## Results

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

The cost is the mean of L2 distance of each of the 70000 images to its nearest facility. The distance is normalized by the number of pixels (coded in the range [0..256])

|  mnist       |  mean entropy  |    cost      |  nb facility | 
|  :---:       |  :---:         |    :---:     |     :---:    |
|   digits     |    0.84        |     2.13     |      75      |
|   fashion    |    0.837       |     1.78     |      75      |
    

### New Fraweworks for Offline and Streaming Coreset Constructions.

We construct coresets with this algorithm.
The points, with their weights attached are then clustered by our simplistic kmedoid algorithm.
The centers obtained and the corresponding clustering cost are then compared with the clustering cost obtained
with the [kmedoids](https://crates.io/crates/kmedoids) crate for L1 distance.  
The computation times given are system time elapsed and total cpu times (to account for parallelism) 


|  mnist       |     cost       |  time(s)     |  time(cpu) |
|  :---:       |  :------:      |    :---:     |  :-------: |
|   digits     |                |              |            |
|   fashion    |                |              |            |



### Facility Location in sublinear time.

this algorithm (at least for this implementation) requires 10s system time and high threading to get similar entropy and costs. 


## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.