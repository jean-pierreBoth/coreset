//! implement trait for iterating on file hnsw.data
//!
#![allow(unused)]

use std::marker::PhantomData;

use indexmap::map::Keys;

use hnsw_rs::datamap::DataMap;

use coreset::makeiter::MakeIter;

/// The structure implementing MakeIter trait for Hnsw data
struct HnswMakeIter<'a, T> {
    datamap: &'a DataMap,
    phantom: PhantomData<T>,
}

impl<'a, T> HnswMakeIter<'a, T> {
    pub fn new(datamap: &'a DataMap, phantom: PhantomData<T>) -> Self {
        HnswMakeIter { datamap, phantom }
    }
}

// our Iterator over hnsw data
struct HnswIter<'a, T> {
    mapref: &'a DataMap,
    keys: Keys<'a, usize, usize>,
    phantom: PhantomData<T>,
}

impl<'a, T> HnswIter<'a, T> {
    pub fn new(mapref: &'a DataMap) -> Self {
        let keys = mapref.get_dataid_iter();
        HnswIter {
            mapref,
            keys,
            phantom: PhantomData,
        }
    }
}

impl<'a, T> Iterator for HnswIter<'a, T>
where
    T: 'a + Clone + Send + Sync + std::fmt::Debug,
{
    type Item = (usize, Vec<T>);

    fn next(&mut self) -> Option<Self::Item> {
        let next_key = self.keys.next();
        if next_key.is_none() {
            return None;
        }
        let next_key = next_key.unwrap();
        let v = self.mapref.get_data::<T>(next_key);
        return Some((*next_key, Vec::<T>::from(v.unwrap())));
    }
}

//================================================================

impl<'a, T> MakeIter for HnswMakeIter<'a, T>
where
    T: 'a + Clone + Send + Sync + std::fmt::Debug,
{
    type Item = (usize, Vec<T>);

    fn makeiter(&self) -> HnswIter<'a, T> {
        let keys = self.datamap.get_dataid_iter();
        let hnswiter = HnswIter::<'a, T>::new(&self.datamap);
        return hnswiter;
    }
}