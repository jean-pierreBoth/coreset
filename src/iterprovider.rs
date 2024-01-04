//! A trait defining how to get an iterator on data.   
//! (using Rust >= 1.75) with RPITIT ( return-position impl Trait in trait)

/// DataType is the type produced by the iterator obtained with the *makeiter* function. 
/// This is typically used when all data cannot be in memory. Then an iterator based on IO can be generated with a structure implementing the trait.   
/// Any algorithm (such as [Coreset1](super::sensitivity::Coreset1)) needing an iterator and more than one pass on data to run can use this trait.  
/// The crate hnsw_rs will provide such an iterator on data stored in hnsw database.
pub trait IterProvider {
    /// The data the iterator will produce
    type DataType;
    /// how to get an iterator
    fn makeiter(&self) -> impl Iterator<Item=Self::DataType>;
}