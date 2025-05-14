
## Results

Examples provided are the standard Mnist-digits and Mnist-fashion.
Results are are obtained with sub-crate mnistcheck.

###  Coreset Construction

The coreset points, with their weights attached are clustered by our simplistic kmedoid algorithm.  
The cost of clustering with these centers is then compared with the cost of clustering the whole original data obtained
with the crate [kmedoids](https://crates.io/crates/kmedoids) using the parallel [par_fastermap](https://docs.rs/kmedoids/0.5.0/kmedoids/fn.par_fasterpam.html).

The computation times, in seconds, given are system time elapsed and total cpu times (to account for parallelism) 


#### Results for coreset construction + basic weighted medoid  (L1 distance) 

The size of the coreset was set 0.11 * the number of data points. We asked 10 clusters.

We give the  cost of clustering the coreset and the cost after dispatching a posteriori the whole original data to the medoids position obtained via coreset clustering.  

The normalized mutual information used is the *sqrt* version see the doc related to the structure Contingency.

As the results are random for the coreset algorithm they are given in the form (mean +-standard deviation) obtained on a sample of 20 computations.  



#### Reference results for medoid computations (L1 distance) with par_fastermap

The timings take into account the computing of the distance matrix (fully multithreaded) which counts for more than 85% of time  
|  mnist  |    cost    | nmi (sqrt) | time(sys) s | time(cpu) s |
| :-----: | :--------: | :--------: | :---------: | :---------: |
| digits  | 1.789 10^6 |    0.38    |     143     |    3800     |
| fashion | 2.181 10^6 |    0.49    |     138     |    3805     |



#### Results with 15 iterations in Kmedoids.

|  mnist  |    cost (coreset)     |   cost (whole data)   | nmi +- sigma | time(sys) s | time(cpu) s |
| :-----: | :-------------------: | :-------------------: | :----------: | :---------: | :---------: |
| digits  | (1.864 +- 0.025) 10^6 | (1.873 +- 0.022) 10^6 |              |     5.6     |     41      |
| fashion | (2.250 +- 0.041) 10^6 | (2.255 +- 0.043) 10^6 |              |     5.5     |     41      |

* The costs , are on the average within 5% above the reference cost obtained by par_fastermap, and consistently within 8% at 2 std deviations. The normalized mutual information 



#### Results with 25 iterations in Kmedoids.


|  mnist  |    cost (coreset)     |   cost (whole data)   |  nmi +- sigma  | time(sys) s | time(cpu) s |
| :-----: | :-------------------: | :-------------------: | :------------: | :---------: | :---------: |
| digits  | (1.868 +- 0.021) 10^6 | (1.876 +- 0.021) 10^6 | 0.348 +- 0.027 |     5.6     |     41      |
| fashion | (2.230 +- 0.026) 10^6 | (2.239 +- 0.022) 10^6 | 0.491 +- 0.027 |     5.5     |     40      |

The results are slightly better in the fashion case with 2.6% of overestimation of the cost compared with par_fastermap. 
The results are, on the average less than 5% above the reference cost obtained by par_fastermap, and within 8% at 3 std deviations.

The Normalized Mutual Information between the coreset classification and the Faster_pam algorithm is:
 - For the Digits case : 0.49 +- 0.035
 - For the Fashion case : 0.73 +- 0.05

We see that the 2 algorithms are related in their classification. They seem to have correlated deviations from ground truth
as their mutual information is greater than their respective mutual information with ground truth.

##### Results on the [$\alpha$,$\beta$] approximation can be found [here](./bmor.md)