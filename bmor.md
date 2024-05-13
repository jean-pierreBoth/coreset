# Results for the Bmor algorithm

### Streaming k-means on well clustered data

#### running in one pass. (~ 130 facilities)

|  mnist  | mean entropy | cost  | nb facility |
| :-----: | :----------: | :---: | :---------: |
| digits  |     0.74     | 20.5  |     170     |
| fashion |     0.77     | 25.7  |     170     |

This algorithm runs on both Mnist data in less than 1s on a i9 laptop.

#### running with post contraction of number of facilities. (~ 70 facilities)

The mean entropy of labels distributions (10 labels) in each cluster found is between 0.82 for the digits example and 0.84 for the fashion example.

The cost is the mean of L2 distance of each of the  0 images to its nearest facility. The distance is normalized by the number of pixels (coded in the range [0..256])

|  mnist  | mean entropy | cost  | nb facility |
| :-----: | :----------: | :---: | :---------: |
| digits  |     0.84     | 21.3  |     75      |
| fashion |    0.837     | 17.8  |     75      |
    
Note : the  *Badoiu, Czumaj, Indyk, Sohler ICALP 2005* algorithm (at least for our implementation) requires 10s system time and high threading to get similar entropy and costs. 