//! implementation of trait MakeIter for Mnist data
//!

use std::iter::Iterator;

use coreset::prelude::*;

pub(crate) struct DataIterator<'a> {
    // we must keep the rank
    rank: usize,
    // we must have an access to data
    images: &'a Vec<Vec<f32>>,
}

impl<'a> DataIterator<'a> {
    pub fn new(images: &'a Vec<Vec<f32>>) -> Self {
        log::debug!("new data iterator size : {}", images.len());
        DataIterator { rank: 0, images }
    }
} // end of impl DataIterator

// We could have chosen Item to be (&Vec<f32>, usize) as we have all data in memory.
// But the coreset algorithms will not in general be able to have all data in memory so we
// must pass real data when algos require data from the iterator.
impl<'a> Iterator for DataIterator<'a> {
    type Item = (usize, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rank < self.images.len() {
            let rank1 = self.rank;
            self.rank += 1;
            return Some((rank1, self.images[rank1].clone()));
        } else {
            return None;
        }
    }
} // end of Iterator for MnistData

/// a structure implementing MakeIter
pub(crate) struct DataForIterator<'a> {
    // we must have an access to data
    images: &'a Vec<Vec<f32>>,
}

impl<'a> DataForIterator<'a> {
    pub fn new(images: &'a Vec<Vec<f32>>) -> Self {
        DataForIterator { images }
    }
} // end of impl DataForIterator

impl<'a> MakeIter for DataForIterator<'a> {
    type Item = (usize, Vec<f32>);
    //
    fn makeiter(&self) -> impl Iterator<Item = <Self as coreset::prelude::MakeIter>::Item> {
        let iterator = DataIterator::new(self.images);
        return iterator;
    }
} //end impl MakeIter
