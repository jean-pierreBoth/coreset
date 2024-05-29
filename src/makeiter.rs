//! A trait defining how to get an iterator on data.   
//! (using Rust >= 1.75) with RPITIT ( return-position impl Trait in trait)

/// This trait defines how the algorithms expect to build an iterator.  
/// It is used when to avoid having all data in memory.
/// It can be an iterator based on IO with a structure implementing the trait.    
///  
///
/// Any algorithm (such as [Coreset1](super::sensitivity::Coreset1)) needing an iterator and more than one pass on data to run must use this trait.  
///   
/// The crate hnsw_rs will provide such an iterator on data stored in hnsw database.
pub trait MakeIter {
    /// an item of data iterated over
    type Item;
    /// how to get an iterator
    fn makeiter(&self) -> impl Iterator<Item = Self::Item>;
}
