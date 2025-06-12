# nmi

Normalized Mutual information is based on the paper:  
    - Vinh.N.X Information Theoretic Measures for clustering comparison. [Vinh 2010](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)


This crate defines various quality measures of clustering based on information theory.
They are normalized with value in the interval [0,1] and some are metrics.  
It is also possible to compare different clustering and to quantify how the affectation to items of 2 algorithms are related. It relies on a contingency table.  
The various measures are described in the [contingency](./src/contingency.rs) file.


We preferentially use *get_nmi_sqrt()*

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.