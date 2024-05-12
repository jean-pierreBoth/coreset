# sub crate fromhnsw

Provides coreset computations from data stored in hnsw structures coming from crate [hsnw_rs](https://crates.io/crates/hnsw_rs).

## Command line

The documentation of the binary **hcore** explains the parameters of the command and ouptut
which has two forms:
  - **hnscore  --dir (-d) dirname  --fname (-f) hnswname  --typename (-t) typename**
  or in its complete form:  
  - **hnscore  --dir (-d) dirname  --fname (-f) hnswname  --typename (-t) typename  clustercore --cluster nbcluster [--beta b] [--gamma g] --**

To get the whole doc:  
**cargo doc --no-deps --all**

## Building

To compile the whole crate run:  
**cargo build --release --workspace --all**


To compile the subcrate *fromhnsw* enabling coreset computations on hnsw data run :  
**cargo build --release --all  --bin hcore**



