//! describes affectation of data to clusters

use dashmap::DashMap;
use std::collections::HashMap;
use std::hash::Hash;

/// The cluster affectation of any clustering scheme should be able to provide a structure implementing this trait.
///
/// Typically an affectation abstract a clusterization as something giving the label or the rank of the cluster attached to a dataid.  
/// The DataId should satisfy the Hash trait. Morally The label is a discrete value (satisfy the PrimInt + Unsigned trait).
///
pub trait Affectation<DataId, DataLabel> {
    /// given a dataId, returns its label or cluster Id
    fn get_affectation(&self, dataid: DataId) -> DataLabel;
    /// returns number of points in clusters
    fn get_nb_points(&self) -> usize;
    /// iterator on couples (dataid, label)
    fn iter(&self) -> impl Iterator<Item = (DataId, DataLabel)>;
}

//==============================================================================

/// Clusters defined by a HashMap
pub struct HashAffectation<DataId, DataLabel> {
    affectation: HashMap<DataId, DataLabel>,
}

impl<DataId, DataLabel> Affectation<DataId, DataLabel> for HashAffectation<DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    fn get_affectation(&self, id: DataId) -> DataLabel {
        *self.affectation.get(&id).unwrap()
    }

    fn get_nb_points(&self) -> usize {
        self.affectation.len()
    }

    fn iter(&self) -> impl Iterator<Item = (DataId, DataLabel)> {
        HashAffectationIter::new(self)
    }
}

//

/// an iterator over affectations
pub struct HashAffectationIter<'a, DataId, DataLabel> {
    _from: &'a HashAffectation<DataId, DataLabel>,
    iter: std::collections::hash_map::Iter<'a, DataId, DataLabel>,
}

impl<'a, DataId, DataLabel> HashAffectationIter<'a, DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    //
    pub fn new(from: &'a HashAffectation<DataId, DataLabel>) -> Self {
        HashAffectationIter {
            _from: from,
            iter: from.affectation.iter(),
        }
    }
}

impl<DataId, DataLabel> Iterator for HashAffectationIter<'_, DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    type Item = (DataId, DataLabel);
    //
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| (*(item.0), *(item.1)))
    }
}

//==============================================================================

/// Clusters defined by a DashMap
pub struct DashAffectation<'a, DataId, DataLabel> {
    affectation: &'a DashMap<DataId, DataLabel>,
}

impl<DataId, DataLabel> Affectation<DataId, DataLabel> for DashAffectation<'_, DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    fn get_affectation(&self, id: DataId) -> DataLabel {
        let item = self.affectation.get(&id).unwrap();
        *item.value()
    }

    fn get_nb_points(&self) -> usize {
        self.affectation.len()
    }

    fn iter(&self) -> impl Iterator<Item = (DataId, DataLabel)> {
        DashAffectationIter::new(self)
    }
}

impl<'a, DataId, DataLabel> DashAffectation<'a, DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    pub fn new(affectation: &'a DashMap<DataId, DataLabel>) -> Self {
        DashAffectation { affectation }
    }
}

//

/// an iterator over affectations defined using a DashMap
pub struct DashAffectationIter<'a, DataId, DataLabel> {
    _from: &'a DashAffectation<'a, DataId, DataLabel>,
    iter: dashmap::iter::Iter<'a, DataId, DataLabel>,
}

impl<'a, DataId, DataLabel> DashAffectationIter<'a, DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    //
    pub fn new(from: &'a DashAffectation<DataId, DataLabel>) -> Self {
        DashAffectationIter {
            _from: from,
            iter: from.affectation.iter(),
        }
    }
}

impl<DataId, DataLabel> Iterator for DashAffectationIter<'_, DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    type Item = (DataId, DataLabel);
    //
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|item| (*(item.key()), *(item.value())))
    }
} // end of DashAffectationIter

//===============================================================================

// Affectation defined by a Vec<DataLabel>

/// Clusters defined by a Vec, DataId is an usize Vec\[i\] gives the label of the i-th data
pub struct VecAffectation<DataLabel> {
    affectation: Vec<DataLabel>,
}

impl<DataLabel> VecAffectation<DataLabel>
where
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    /// builds a vector affectation
    pub fn new(affectation: Vec<DataLabel>) -> Self {
        VecAffectation { affectation }
    }
}

impl<DataLabel> Affectation<usize, DataLabel> for VecAffectation<DataLabel>
where
    DataLabel: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
{
    fn get_affectation(&self, id: usize) -> DataLabel {
        self.affectation[id]
    }

    fn get_nb_points(&self) -> usize {
        self.affectation.len()
    }

    fn iter(&self) -> impl Iterator<Item = (usize, DataLabel)> {
        (0..self.affectation.len())
            .zip(self.affectation.iter())
            .map(|it| (it.0, *(it.1)))
    }
}
