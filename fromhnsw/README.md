# fromhnsw

Provides coreset computations from data stored in hnsw structures coming from crate [hsnw_rs](https://crates.io/crates/hnsw_rs).

## Parameters


## Command line

## Building

To compile main crate run:  
**cargo build --release --workspace --all**


To compile the subcrate *fromhnsw* enabling coreset computations on hnsw data run :  
**cargo build --release --all**

Doc building of the library

**cargo doc --no-deps  --all**

Building doc of the whole workspace and binary 

**cargo doc --no-deps  --workspace --all  --open --bin hcore**

## Extensions
