//! This module implements various mutual information from contingency table computation

use indexmap::IndexSet;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::hash::Hash;

use num_traits::int::PrimInt;
use std::marker::PhantomData;

use super::affect::Affectation;

//================================================================================

#[cfg_attr(doc, katexit::katexit)]
/// Contingency table associated to the 2 clusterization (affectations) to compare.  
/// We can compare either an algorithm to reference) labels of data or 2 clusters algorithms.  
/// The various merit functions relies on comparisons entropy of cluster distribution.  
///
///
/// We contruct a contingency matrix $ \left(n_{ij} \right) $  with $ i \le n_{1}, j \le n_{2} $ with $ n_{ij} =   | C_{1}[i] \cap C_{2}[j] | $ with $ C_{1}[i] $ designing the i-th cluster in C1 clusterization.
/// We call:
/// - N : the number of elements to cluster
/// - $NC_{1}$ (resp. $NC_{2}$) the number of clusters of the first (resp. second) clusterization.
///
/// The following entropies are then computed:
/// - $ H(C_{1}) = - \sum_{i \le NC_{1}}  \frac{|C_1[i]|}{N} \log \frac{|C_1[i]|}{N} $
/// - $ H(C_{2}) = - \sum_{i \le NC_{2}}  \frac{|C_2[i]|}{N} \log \frac{|C_2[i]|}{N} $  
/// - $ H(C_{1},C_{2}) = - \sum_{i \le NC_{1}, j \le NC_{2}}  \frac{n_{ij}}{N} \log \frac{n_{ij}}{N} $
/// - $ H(C_{1}| C_{2}) = - \sum_{i \le NC_{1}, j \le NC_{2}}  \frac{n_{ij}}{N} \log \frac{n_{ij}/N} {|C_2[j]|/N} $
/// - $ I(C_{1}| C_{2}) = \sum_{i \le NC_{1}, j \le NC_{2}}  \frac{n_{ij}}{N} \log \frac{n_{ij}/N} { |C_1[i]| *|C_2[j]|/N^{2}} $.  
///
/// Various indicators can then be computed (some are even metrics), we choose normalized versions i.e their values are in the range [0,1].  
/// See the different functions
pub struct Contingency<Clusterization, DataId, DataLabel>
where
    Clusterization: Affectation<DataId, DataLabel>,
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    // clusters (or reference)
    clusters1: Clusterization,
    // clusters (or reference)
    #[allow(unused)]
    clusters2: Clusterization,
    // transform labels set to usize range for array indexation
    #[allow(unused)]
    labels1: IndexSet<DataLabel>,
    #[allow(unused)]
    labels2: IndexSet<DataLabel>,
    // The contingency table. dimension (cluster1.nb_cluster, cluster2.nb_cluster)
    #[allow(unused)]
    table: Array2<usize>,
    // number of elements in each clusters of cluster1
    #[allow(unused)]
    c1_size: Array1<usize>,
    // number of elements in each clusters of cluster2
    #[allow(unused)]
    c2_size: Array1<usize>,
    // entropies and information
    entropy_1: f64,
    entropy_2: f64,
    entropy_12: f64,
    entropy_1cond2: f64,
    information_12: f64,
    //
    _t_id: PhantomData<DataId>,
    _t_label: PhantomData<DataLabel>,
}

impl<DataId, DataLabel, Clusterization> Contingency<Clusterization, DataId, DataLabel>
where
    Clusterization: Affectation<DataId, DataLabel>,
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt + Hash,
{
    /// **The first (resp. second) argument will be used as rows (resp. columns) of the contingency matrix**
    pub fn new(clusters1: Clusterization, clusters2: Clusterization) -> Self {
        assert_eq!(clusters1.get_nb_points(), clusters2.get_nb_points());
        //
        log::debug!("entering Contingency::new");
        //
        // converts labels to contiguous interval of usize. label_rank = IndexSet::get_index_of(label).unwrap()
        //
        let mut labels1 = IndexSet::<DataLabel>::with_capacity(50);
        let affect1_iter = clusters1.iter();
        for (_, label) in affect1_iter {
            labels1.insert(label);
        }
        let mut labels2 = IndexSet::<DataLabel>::with_capacity(50);
        let affect2_iter = clusters2.iter();
        for (_, label) in affect2_iter {
            labels2.insert(label);
        }
        let nb_labels1 = labels1.len();
        let nb_labels2 = labels2.len();
        let mut table = Array2::<usize>::zeros((nb_labels1, nb_labels2));
        let mut c1_size: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>> =
            Array1::<usize>::zeros(nb_labels1);
        let mut c2_size: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>> =
            Array1::<usize>::zeros(nb_labels2);
        // computes contingency table
        let affect1_iter = clusters1.iter();
        // we loop on affect1_iter, query each item relativeley to clusters2 and dispatch to table
        for (id1, label1) in affect1_iter {
            // we loop on affect2_iter, query each item relativeley to clusters2 and dispatch to table
            let rank_l1 = labels1.get_index_of(&label1).unwrap();
            c1_size[rank_l1] += 1;
            let label2 = clusters2.get_affectation(id1);
            let rank_l2 = labels2.get_index_of(&label2).unwrap();
            c2_size[rank_l2] += 1;
            // summing on columns each item in cluster1 appears exactly once
            // and summing on rows item in cluster2 appears exactly once (as long as the same set of DataId is in both clusterization)
            table[[rank_l1, rank_l2]] += 1;
        }
        log::debug!("contingency table computed ({},{})", nb_labels1, nb_labels2);
        // compute entropies H and I
        let nb_total_usize = c1_size.iter().fold(0, |acc, x| acc + *x);
        assert_eq!(nb_total_usize, clusters1.get_nb_points());
        assert_eq!(nb_total_usize, c2_size.iter().fold(0, |acc, x| acc + *x));
        let nb_total = nb_total_usize as f64;
        let entropy_1 = c1_size.iter().fold(0., |acc, x| {
            acc - *x as f64 * log_with0(*x as f64 / nb_total)
        }) / nb_total;
        //
        let entropy_2 = c2_size.iter().fold(0., |acc, x| {
            acc - *x as f64 * log_with0(*x as f64 / nb_total)
        }) / nb_total;
        //
        let entropy_12 = table.iter().fold(0., |acc, x| {
            acc - *x as f64 * log_with0(*x as f64 / nb_total)
        }) / nb_total;
        //
        let mut entropy_1cond2: f64 = 0.;
        let mut information_12: f64 = 0.;
        for i in 0..nb_labels1 {
            let frac_i: f64 = c1_size[i] as f64 / nb_total;
            for j in 0..nb_labels2 {
                let frac_ij = table[[i, j]] as f64 / nb_total;
                let frac_j: f64 = c2_size[j] as f64 / nb_total;
                //
                entropy_1cond2 -= table[[i, j]] as f64 * log_with0(frac_ij / frac_j);
                information_12 += table[[i, j]] as f64 * log_with0((frac_ij) / (frac_i * frac_j));
            }
        }
        entropy_1cond2 /= nb_total;
        information_12 /= nb_total;
        //
        log::debug!("Contingency allocation");
        //
        Contingency {
            clusters1,
            clusters2,
            labels1,
            labels2,
            table,
            c1_size,
            c2_size,
            entropy_1,
            entropy_2,
            entropy_12,
            entropy_1cond2,
            information_12,
            _t_id: PhantomData,
            _t_label: PhantomData,
        }
    }

    /// returns entropy of cluster 1 distribution
    pub fn get_entropy_1(&self) -> f64 {
        self.entropy_1
    }

    /// returns entropy of cluster 1 distribution
    pub fn get_entropy_2(&self) -> f64 {
        self.entropy_2
    }

    /// returns joint entropy of clusters distribution
    pub fn get_joint_entropy(&self) -> f64 {
        self.entropy_12
    }

    pub fn get_entropy_1cond2(&self) -> f64 {
        self.entropy_1cond2
    }

    pub fn get_information(&self) -> f64 {
        self.information_12
    }

    /// logs the various entropies computed
    pub fn dump_entropies(&self) {
        log::info!(" entropy1 : {:.3e}", self.entropy_1);
        log::info!(" entropy2 : {:.3e}", self.entropy_2);
        log::info!(" entropy_12 : {:.3e}", self.entropy_12);
        log::info!(" entropy_1cond2 : {:.3e}", self.entropy_1cond2);
        log::info!(
            " information_12: {:.3e} expecation upper bound {:.3e}",
            self.information_12,
            self.information_ebound()
        );
    }

    /// upper bound of expectation of joint information
    pub fn information_ebound(&self) -> f64 {
        let nbclust1 = self.c1_size.len();
        let nbclust2: usize = self.c2_size.len();
        let nb_total = self.clusters1.get_nb_points();
        let a: usize = nb_total + nbclust1 * nbclust2 - nbclust1 - nbclust2;
        //
        ((a as f64) / (nb_total - 1) as f64).ln()
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// compute normalized mutual information joint version.  
    /// returns $ \frac{I(C_{1}, C_{2})}{H(C_{1}, C_{2})} $.   
    /// Note that : $ 1. - \frac{I(C_{1}, C_{2})}{H(C_{1}, C_{2})} $ is a metric.
    pub fn get_nmi_joint(&self) -> f64 {
        self.information_12 / self.entropy_12
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// compute normalized mutual information max version
    /// returns $ \frac{I(C_{1}, C_{2})}{max (H(C_{1}),  H(C_{2})} $.    
    /// Note that : $ .1 - \frac{I(C_{1}, C_{2})}{max (H(C_{1}),  H(C_{2})} $ is a metric.
    pub fn get_nmi_max(&self) -> f64 {
        self.information_12 / (self.entropy_1.max(self.entropy_2))
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// compute normalized mutual information mean version
    /// returns $ \frac{I(C_{1}, C_{2})}{0.5 * (H(C_{1}) +  H(C_{2})} $.    
    /// Note that : $ .1 - \frac{I(C_{1}, C_{2})}{max (H(C_{1}),  H(C_{2})} $ is not a metric.
    pub fn get_nmi_mean(&self) -> f64 {
        2. * self.information_12 / (self.entropy_1 + self.entropy_2)
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// compute normalized mutual information sqrt version
    /// returns $ \frac{I(C_{1}, C_{2}) }{\sqrt{ H(C_{1}) * H(C_{2})} } $   
    /// Note that : $ 1. - \frac{I(C_{1}, C_{2})}{\sqrt{H(C_{1}) * H(C_{2})} } $ is not a metric.
    /// The function logs as info lower bound of the adjusted value. For large number of data, the correction is negligible
    pub fn get_nmi_sqrt(&self) -> f64 {
        let nmi = self.information_12 / (self.entropy_1 * self.entropy_2).sqrt();
        let iup = self.information_12_expectation_upper();
        let ami = (self.information_12 - iup) / ((self.entropy_1 * self.entropy_2).sqrt() - iup);
        log::info!("adjusted nmi_sqrt will be greater than : {:.3e}", ami);
        nmi
    }

    /// computes upper bound for cross informaion, valid for large n
    /// See Th 7 from [Vinh 2010](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)
    pub fn information_12_expectation_upper(&self) -> f64 {
        let n: usize = self.clusters1.get_nb_points() + self.c1_size.len() * self.c2_size.len()
            - self.c1_size.len()
            - self.c2_size.len();
        let d: usize = self.clusters1.get_nb_points() - 1;
        (n as f64 / d as f64).ln()
    }

    /// returns for now an upper bound
    pub fn information_12_expectation(&self) -> f64 {
        let bound = self.information_12_expectation_upper();
        log::info!("an upper bound is {:.3e}", bound);
        log::info!("not yet implemented");
        bound
    }

    // methods to get entropies by row

    /// return entropy of distribution of items in clust of first (row) clusterization along the second (columns) clusterization
    /// The purpose is to find which clusters of the first Clusterization are distributed with less incertitude
    pub fn get_row_entropy(&self, row: usize) -> f64 {
        let mut entropy_cond1: f64 = 0.;
        //
        let nb_total = self.c1_size[row];
        let (_, nb_column) = self.table.dim();
        for j in 0..nb_column {
            let frac_ij = self.table[[row, j]] as f64 / nb_total as f64;
            //
            entropy_cond1 -= self.table[[row, j]] as f64 * log_with0(frac_ij);
        }
        //
        entropy_cond1 / nb_total as f64
    } // end of get_row_entropy

    /// return entropy of distribution of items in clust of second (col) clusterization along the first (rows) clusterization
    /// The purpose is to find which clusters of the first Clusterization are distributed with less incertitude
    pub fn get_column_entropy(&self, col: usize) -> f64 {
        let mut entropy_cond2: f64 = 0.;
        //
        let nb_total = self.c2_size[col];
        let (nb_row, _) = self.table.dim();
        for i in 0..nb_row {
            let frac_ij = self.table[[i, col]] as f64 / nb_total as f64;
            //
            entropy_cond2 -= self.table[[i, col]] as f64 * log_with0(frac_ij);
        }
        //
        entropy_cond2 / nb_total as f64
    }

    /// collect row entropies for all rows (fist clusterization)
    pub fn get_row_entropies(&self) -> Vec<f64> {
        let (nb_row, _) = self.table.dim();
        (0..nb_row)
            .map(|i| self.get_row_entropy(i))
            .collect::<Vec<f64>>()
    }

    /// collect column entropies for all columns (second clusterization)
    pub fn get_col_entropies(&self) -> Vec<f64> {
        let (_, nb_col) = self.table.dim();
        (0..nb_col)
            .map(|j| self.get_column_entropy(j))
            .collect::<Vec<f64>>()
    }

    pub fn get_row(&self, row: usize) -> ArrayView1<usize> {
        self.table.row(row)
    }

    pub fn get_col(&self, col: usize) -> ArrayView1<usize> {
        self.table.column(col)
    }

    pub fn get_dim(&self) -> (usize, usize) {
        self.table.dim()
    }

    pub fn get_table(&self) -> ArrayView2<usize> {
        self.table.view()
    }

    /// return a vector correspondance between rank of matrix and Labels.
    /// I.e the k entry of the vector gives the label of the row (column) k.  
    /// - 0 is row dimension (ndarray conventions)
    /// - 1 is column
    ///
    pub fn get_labels_rank(&self, dim: usize) -> Vec<DataLabel> {
        let labels_iter = match dim {
            0 => self.clusters1.iter(),
            1 => self.clusters2.iter(),
            _ => {
                panic!("dim must 0 or 1 for row or column dimension");
            }
        }; //end match
        let mut labels_set = IndexSet::<DataLabel>::with_capacity(50);
        for (_data_id, label) in labels_iter {
            labels_set.insert(label);
        }
        // convert to a vector giving label in function of row or column
        let mut to_labels = Vec::<DataLabel>::with_capacity(labels_set.len());
        for (_rank, l) in labels_set.iter().enumerate() {
            to_labels.push(*l);
        }
        to_labels
    } // end of get_labels_rank
} // end of Contingency

// for entropy calculations log(0) = 0...
fn log_with0(arg: f64) -> f64 {
    if arg > 0. {
        arg.ln()
    } else if arg < 0. {
        panic!("log cannot have negative arg");
    } else {
        0.
    }
}
