# fromhnsw

Provides coreset computations from data stored in hnsw structures coming from crate [hsnw_rs](https://crates.io/crates/hnsw_rs).

## Parameters

The coreset command takes as arguments (they are explained in detail in Bmor documentation):
 - beta:  defaults to 2.
 - gamma: defaults to 2. Increasing it allocates a greater number of facilites.

## Command line

## Building

To compile the whole crate run:  
**cargo build --release --workspace --all**


To compile the subcrate *fromhnsw* enabling coreset computations on hnsw data run :  
**cargo build --release --all  --bin hcore**

Doc building of the library

**cargo doc --no-deps  --all**

Building doc of the whole workspace and binary 

**cargo doc --no-deps  --workspace --all   [--open] --bin hcore**

## Extensions
