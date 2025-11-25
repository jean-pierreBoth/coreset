//! utility for Mnist
//!
//!
//!

use anyhow::anyhow;
use ndarray::{s, Array1, Array2, Array3, Axis};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;

/// This structure can load MNIST data either in CSV format or in compressed format [MNIST idx](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
/// stores labels (i.e : digits between 0 and 9) coming from file train-labels-idx1-ubyte      
/// and hand written characters as 28*28 images with values between 0 and 255 coming from train-images-idx3-ubyte
/// images\[i,j,k\] stores pixel (i,j) of data item k
pub struct MnistData {
    _image_filename: PathBuf,
    _label_filename: PathBuf,
    pub(crate) images: Array3<u8>,
    pub(crate) labels: Array1<u8>,
}

impl MnistData {
    pub fn new(image_filename: PathBuf, label_filename: PathBuf) -> std::io::Result<MnistData> {
        let image_path = PathBuf::from(image_filename.clone());
        let image_file = OpenOptions::new().read(true).open(image_path)?;
        let mut image_io = BufReader::new(image_file);
        let images = read_image_file(&mut image_io);
        // labels
        let label_path = PathBuf::from(label_filename.clone());
        let labels_file = OpenOptions::new().read(true).open(label_path)?;
        let mut labels_io = BufReader::new(labels_file);
        let labels = read_label_file(&mut labels_io);
        Ok(MnistData {
            _image_filename: image_filename,
            _label_filename: label_filename,
            images,
            labels,
        })
    } // end of new for MnistData

    /// load data from csv files (train or test)
    pub fn new_from_csv(image_filename: PathBuf) -> std::io::Result<MnistData> {
        let image_path = PathBuf::from(image_filename.clone());
        let image_file = OpenOptions::new().read(true).open(&image_path)?;
        let mut image_io = BufReader::new(image_file);
        let (labels, images) = read_image_csv(&mut image_io).unwrap();
        Ok(MnistData {
            _image_filename: image_path.clone(),
            _label_filename: PathBuf::from(""),
            images,
            labels,
        })
    } // end of new_from_csv for MnistData

    /// returns labels of images. lables\[k\] is the label of the k th image.
    pub fn get_labels(&self) -> &Array1<u8> {
        &self.labels
    }

    /// returns images. images are stored in Array3 with Array3[[.., .., k]] being the k images!
    /// Each image is stored as it is in the Mnist files, Array3[[i, .., k]] is the i row of the k image
    pub fn get_images(&self) -> &Array3<u8> {
        &self.images
    }
} // end of impl MnistData

/// read from idx format
pub fn read_image_file(io_in: &mut dyn Read) -> Array3<u8> {
    // read 4 bytes magic
    // to read 32 bits in network order!
    let toread: u32 = 0;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let magic = u32::from_be(toread);
    assert_eq!(magic, 2051);
    // read nbitems
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbitem = u32::from_be(toread);
    assert!(nbitem == 60000 || nbitem == 10000);
    //  read nbrow
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbrow = u32::from_be(toread);
    assert_eq!(nbrow, 28);
    // read nbcolumns
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbcolumn = u32::from_be(toread);
    assert_eq!(nbcolumn, 28);
    // for each item, read a row of nbcolumns u8
    let mut images: ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 3]>> =
        Array3::<u8>::zeros((nbrow as usize, nbcolumn as usize, nbitem as usize));
    let mut datarow = vec![0u8; nbcolumn as usize];
    for k in 0..nbitem as usize {
        for i in 0..nbrow as usize {
            let it_slice = datarow.as_mut_slice();
            io_in.read_exact(it_slice).unwrap();
            let mut smut_ik = images.slice_mut(s![i, .., k]);
            assert_eq!(nbcolumn as usize, it_slice.len());
            assert_eq!(nbcolumn as usize, smut_ik.len());
            for j in 0..smut_ik.len() {
                smut_ik[j] = it_slice[j];
            }
            //    for j in 0..nbcolumn as usize {
            //        *(images.get_mut([i,j,k]).unwrap()) = it_slice[j];
            //   }
            // how do a block copy from read slice to view of images.
            // images.slice_mut(s![i as usize, .. , k as usize]).assign(&Array::from(it_slice)) ;
        }
    }
    images
} // end of readImageFile

/// read images in Csv form and returns Array1 of labels and images in Array3\<u8\>.
/// pixel i,j of the k'th images is stored in index \[i,j,k\]
pub fn read_image_csv(bufreader: &mut dyn Read) -> anyhow::Result<(Array1<u8>, Array3<u8>)> {
    //
    let mut num_record: usize = 0;

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(bufreader);
    //
    let nb_fields = 785;
    let nb_row: usize = 28;
    let nb_column: usize = 28;
    //
    let mut labels: Vec<u8> = Vec::<u8>::with_capacity(10000);
    let mut images: Array3<u8> = Array3::<u8>::zeros((nb_row as usize, nb_column as usize, 0));
    //
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        num_record += 1;
        //
        let record = result?;
        if record.len() != nb_fields {
            println!("record {} record has {} fields", num_record, record.len());
            return Err(anyhow!(
                "record {} record has {} fields",
                num_record,
                record.len()
            ));
        }
        let mut new_image = Array2::<u8>::zeros((nb_row, nb_column));
        for k in 0..nb_fields {
            let field = record.get(k).unwrap();
            // decode into Ix type
            if k == 0 {
                if let std::result::Result::Ok(val) = field.parse::<u8>() {
                    labels.push(val);
                } else {
                    log::debug!("error decoding field  of record {}", num_record);
                    return Err(anyhow!("error decoding field 1of record  {}", num_record));
                }
            } else {
                let row = (k - 1) / 28;
                let column = (k - 1) % 28;
                if let std::result::Result::Ok(val) = field.parse::<u8>() {
                    new_image[[row, column]] = val;
                } else {
                    log::debug!("error decoding field  of record {}", num_record);
                    return Err(anyhow!("error decoding field 1of record  {}", num_record));
                }
            }
        } // end for k
        images.push(Axis(2), new_image.view()).unwrap();
    }
    //
    assert!(num_record == 10000 || num_record == 60000);
    assert_eq!(images.ndim(), 3);
    assert_eq!(num_record, images.dim().2);
    log::info!("number of records loaded : {:?}", images.dim().2);
    //
    Ok((Array1::<u8>::from_vec(labels), images))
}

//

pub fn read_label_file(io_in: &mut dyn Read) -> Array1<u8> {
    // to read 32 bits in network order!
    let toread: u32 = 0;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let magic = u32::from_be(toread);
    assert_eq!(magic, 2049);
    // read nbitems
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbitem = u32::from_be(toread);
    assert!(nbitem == 60000 || nbitem == 10000);
    let mut labels_vec = vec![0u8; nbitem as usize];
    io_in.read_exact(&mut labels_vec).unwrap();
    Array1::from(labels_vec)
} // end of fn read_label

//=====================================================================

// read from idx*byte files from a directory and returns images as f32 normalized by number of pixels
pub fn io_from_non_csv(not_csv_dir: &str) -> anyhow::Result<(Vec<u8>, Vec<Vec<f32>>)> {
    let mut images_as_v: Vec<Vec<f32>> = Vec::<Vec<f32>>::with_capacity(70000);
    let mut labels: Vec<u8> = Vec::<u8>::new();

    let fnames = [
        ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    ];
    // one pass on train data, one pass on test data
    for i in 0..fnames.len() {
        let mut image_path = PathBuf::from(not_csv_dir);
        image_path.push(fnames[i].0);
        let image_file_res = OpenOptions::new().read(true).open(&image_path);
        if image_file_res.is_err() {
            println!("could not open image file : {:?}", image_path);
            return Err(anyhow!("could not open image file : {:?}", image_path));
        }
        let mut label_path = PathBuf::from(not_csv_dir);
        label_path.push(fnames[i].1);
        let label_file_res = OpenOptions::new().read(true).open(&label_path);
        if label_file_res.is_err() {
            println!("could not open label file : {:?}", label_path);
            return Err(anyhow!("could not open label file : {:?}", label_path));
        }

        assert_eq!(images_as_v.len(), labels.len());
        log::info!("io_from_non_csv loaded {}", images_as_v.len());
        //
        let mnist_data = MnistData::new(image_path.clone(), label_path.clone()).unwrap();
        let images = mnist_data.get_images();
        labels.append(&mut mnist_data.get_labels().to_vec());
        let (_, _, nbimages) = images.dim();
        //
        for k in 0..nbimages {
            let v: Vec<f32> = images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32 / (28. * 28.))
                .collect();
            images_as_v.push(v);
        }
    } // drop mnist_data
      //
    assert_eq!(labels.len(), images_as_v.len());
    log::info!("o_from_non_csv loaded {}", images_as_v.len());
    //
    return Ok((labels, images_as_v));
}

//

// read data from a directory containing files mnist_train.csv and mnist_test.csv and returns images as f32 normalized by number of pixels
pub fn io_from_csv(not_csv_dir: &str) -> anyhow::Result<(Vec<u8>, Vec<Vec<f32>>)> {
    let mut images_as_v: Vec<Vec<f32>> = Vec::<Vec<f32>>::new();
    let mut labels: Vec<u8> = Vec::new();

    let names = ["mnist_train.csv", "mnist_test.csv"];

    for name in names {
        let mut image_path = PathBuf::from(not_csv_dir);
        image_path.push(name);
        let image_file_res = OpenOptions::new().read(true).open(&image_path);
        if image_file_res.is_err() {
            println!("could not open image file : {:?}", image_path);
            return Err(anyhow!("could not open image file : {:?}", image_path));
        }
        //
        {
            let mnist_data = MnistData::new_from_csv(image_path.clone()).unwrap();
            let images = mnist_data.get_images();
            labels.append(&mut mnist_data.get_labels().to_vec());
            let (_, _, nbimages) = images.dim();
            //
            for k in 0..nbimages {
                let v: Vec<f32> = images
                    .slice(s![.., .., k])
                    .iter()
                    .map(|v| *v as f32 / (28. * 28.))
                    .collect();
                images_as_v.push(v);
            }
        } // drop mnist_data
    } // end of for on names
    assert_eq!(images_as_v.len(), labels.len());
    log::info!("io_from_csv loaded {}", images_as_v.len());
    //
    Ok((labels, images_as_v))
}

#[cfg(test)]
mod tests {

    use super::*;
    const MNIST_DIGITS_DIR_NOT_CSV: &str = "/home/jpboth/Data/ANN/MNIST";

    // test and compare some values obtained with Julia loading

    #[test]

    fn test_load_mnist_digits() {
        let mut image_path = PathBuf::from(MNIST_DIGITS_DIR_NOT_CSV);
        image_path.push("train-images-idx3-ubyte");
        let image_file_res = OpenOptions::new().read(true).open(&image_path);
        if image_file_res.is_err() {
            println!(
                "test_load_mnist_digits in io.rs could not open image file : {:?}",
                image_path
            );
            return;
        }

        let mut label_path = PathBuf::from(MNIST_DIGITS_DIR_NOT_CSV);
        label_path.push("train-labels-idx1-ubyte");
        let label_file_res = OpenOptions::new().read(true).open(&label_path);
        if label_file_res.is_err() {
            println!(
                "test_load_mnist_digits in io.rs could not open label file : {:?}",
                label_path
            );
            return;
        }

        let mnist_data = MnistData::new(image_path, label_path).unwrap();
        let labels = mnist_data.get_labels();
        let images = mnist_data.get_images();
        let nblabels = labels.len();
        assert_eq!(0x3c, *images.get([9, 14, 9]).unwrap());
        assert_eq!(0xfd, images[(14, 9, 9)]);
        // check some value of the tenth images

        // check first and last labels
        assert_eq!(5, labels[0]);
        assert_eq!(8, labels[nblabels - 1]);
        assert_eq!(1, 1);
    } // end test_load
} // end module tests
