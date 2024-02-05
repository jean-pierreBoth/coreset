# Results for the Bmor algorithm

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
    
Note : the  *Badoiu, Czumaj, Indyk, Sohler ICALP 2005* algorithm (at least for our implementation) requires 10s system time and high threading to get similar entropy and costs. 