//! Structure and functions to read MNIST fashion database
//! To run the examples change the line :  
//! 
//! const MNIST_FASHION_DIR : &'static str = "/home.1/jpboth/Data/Fashion-MNIST/";
//! 
//! command : mnist_fashion  --algo imp or bmor.
//! 
//! The data can be downloaded in the same format as the FASHION database from:  
//! 
//! <https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion>
//! 


use std::io::prelude::*;
use std::io::BufReader;
use ndarray::{Array3, Array1, s};
use std::fs::OpenOptions;
use std::path::PathBuf;


use std::io::Cursor;
use byteorder::{BigEndian, ReadBytesExt};


use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use hnsw_rs::prelude::*;


/// A struct to load/store for Fashion Mnist in the same format as [MNIST data](http://yann.lecun.com/exdb/mnist/)  
/// stores labels (i.e : FASHION between 0 and 9) coming from file train-labels-idx1-ubyte      
/// and objects as 28*28 images with values between 0 and 255 coming from train-images-idx3-ubyte
pub struct MnistData {
    _image_filename : String,
    _label_filename : String,
    images : Array3::<u8>,
    labels : Array1::<u8>,
}


impl MnistData {
    pub fn new(image_filename : String, label_filename : String) -> std::io::Result<MnistData> {
        let image_path = PathBuf::from(image_filename.clone());
        let image_file = OpenOptions::new().read(true).open(&image_path)?;
        let mut image_io = BufReader::new(image_file);
        let images = read_image_file(&mut image_io);
        // labels
        let label_path = PathBuf::from(label_filename.clone());
        let labels_file = OpenOptions::new().read(true).open(&label_path)?;
        let mut labels_io = BufReader::new(labels_file);
        let labels = read_label_file(&mut labels_io);
        Ok(MnistData{
            _image_filename : image_filename,
            _label_filename : label_filename,
            images,
            labels
        } )
    } // end of new for MnistData

    /// returns labels of images. lables\[k\] is the label of the k th image.
    pub fn get_labels(&self) -> &Array1::<u8> {
        &self.labels
    }

    /// returns images. images are stored in Array3 with Array3[[.., .., k]] being the k images!
    /// Each image is stored as it is in the Mnist files, Array3[[i, .., k]] is the i row of the k image
    pub fn get_images(&self) -> &Array3::<u8> {
        &self.images
    }
} // end of impl MnistData



pub fn read_image_file(io_in: &mut dyn Read) -> Array3::<u8> {
    // read 4 bytes magic
    // to read 32 bits in network order!
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let magic = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 2051);
    // read nbitems
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbitem =  Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert!(nbitem == 60000 || nbitem == 10000);
    //  read nbrow
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbrow = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert_eq!(nbrow, 28);   
    // read nbcolumns
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbcolumn = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();     
    assert_eq!(nbcolumn,28);   
    // for each item, read a row of nbcolumns u8
    let mut images = Array3::<u8>::zeros((nbrow as usize , nbcolumn as usize, nbitem as usize));
    let mut datarow = Vec::<u8>::new();
    datarow.resize(nbcolumn as usize, 0);
    for k in 0..nbitem as usize {
        for i in 0..nbrow as usize {
            let it_slice ;
            it_slice = datarow.as_mut_slice();
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



pub fn read_label_file(io_in: &mut dyn Read) -> Array1<u8>{
    // to read 32 bits in network order!
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let magic = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();     
    assert_eq!(magic, 2049);
    // read nbitems
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbitem = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap(); 
    assert!(nbitem == 60000 || nbitem == 10000);
    let mut labels_vec = Vec::<u8>::new();
    labels_vec.resize(nbitem as usize, 0);
    io_in.read_exact(&mut labels_vec).unwrap();
    let labels = Array1::from(labels_vec);
    labels
}  // end of fn read_label

//============================================================================================

pub struct MnistParams {
    algo : Algo
} // end of MnistParams

impl MnistParams {
    pub fn new(algo : Algo) -> Self {
        MnistParams{algo}
    }
    //
    pub fn get_algo(&self) -> Algo { self.algo}
}

fn marrupaxton<Dist : Distance<f32> + Sync + Send + Clone>(_params :&MnistParams, images : &Vec<Vec<f32>>, labels : &Vec<u8>, distance : Dist) {
    //
    let mpalgo = MettuPlaxton::<f32,Dist>::new(&images, distance);
    let alfa = 0.75;
    let mut facilities = mpalgo.construct_centers(alfa);
    //
    let (entropies, labels_distribution) = facilities.dispatch_labels(&images , labels, None);
    //
    let nb_facility = facilities.len();
    for i in 0..nb_facility {
        let facility = facilities.get_facility(i).unwrap();
        log::info!("\n\n facility : {:?}, entropy : {:.3e}", i, entropies[i]);
        facility.read().log();
        let map = &labels_distribution[i];
        for (key, val) in map.iter() {
            println!("key: {key} val: {val}");
        }
    }
    //
    mpalgo.compute_distances(&mut facilities, &images);
}

//========================================================


fn bmor<Dist : Distance<f32> + Sync + Send + Clone>(_params :&MnistParams, images : &Vec<Vec<f32>>, labels : &Vec<u8>, distance : Dist) {
    //
    // if gamma increases, number of facilities increases.
    // if beta increases , upper bound on cost increases faster so the number of phases decreases
    let beta = 2.;
    let gamma = 2.;
    let mut bmor_algo: Bmor<f32, Dist> = Bmor::new(10, 70000, beta, gamma, distance);
    //
    let res = bmor_algo.process_data(images);
    if res.is_err() {
        std::panic!("bmor failed");
    }
    //
    // do we ask for a supplementary contraction pass
    let contraction = false;
    let mut facilities = bmor_algo.end_data(contraction);
    //
    let (entropies, labels_distribution) = facilities.dispatch_labels(&images , labels, None);
    //
    let nb_facility = facilities.len();
    for i in 0..nb_facility {
        let facility = facilities.get_facility(i).unwrap();
        log::info!("\n\n facility : {:?}, entropy : {:.3e}", i, entropies[i]);
        facility.read().log();
        let map = &labels_distribution[i];
        for (key, val) in map.iter() {
            println!("key: {key} val: {val}");
        }
    }
    //
    facilities.cross_distances();
}

//========================================================

pub fn parse_cmd(matches : &ArgMatches) -> Result<MnistParams, anyhow::Error> {
    log::debug!("in parse_cmd");
    if matches.contains_id("algo") {
        println!("decoding argument algo");
        let algoname = matches.get_one::<String>("algo").expect("");
        log::debug!(" got algo : {:?}", algoname);
        match algoname.as_str() {
            "imp" => {
                let params = MnistParams::new(Algo::IMP);
                return Ok(params);
            },
            "bmor" => {
                let params = MnistParams::new(Algo::BMOR);
                return Ok(params);
            }
            //
            _           => {
                log::error!(" algo must be imp or bmor");
                std::process::exit(1);
            }
        }
    }
    //
    return Err(anyhow::anyhow!("bad command"));
} // end of parse_cmd



//========================================================

use clap::{Arg, ArgMatches, ArgAction, Command};


use coreset::prelude::*;

const MNIST_FASHION_DIR : &'static str = "/home/jpboth/Data/ANN/Fashion-MNIST/";

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    log::info!("running mnist_fashion");
    //
    let matches = Command::new("mnist_fashion")
    //        .subcommand_required(true)
            .arg_required_else_help(true)
            .arg(Arg::new("algo")
                .required(true)
                .long("algo")    
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting a algo option imp, bmor "))
        .get_matches();
    //
    let mnist_params = parse_cmd(&matches).unwrap();
    //
    let mut image_fname = String::from(MNIST_FASHION_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_FASHION_DIR);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    let mut images_as_v:  Vec::<Vec<f32>>;
    let mut labels :  Vec<u8>;
    {
        let mnist_train_data  = MnistData::new(image_fname, label_fname).unwrap();
        let images = mnist_train_data.get_images();
        labels = mnist_train_data.get_labels().to_vec();
        let( _, _, nbimages) = images.dim();
        //
        images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        for k in 0..nbimages {
            // we convert to float normalized 
            let v : Vec<f32> = images.slice(s![.., .., k]).iter().map(|v| *v as f32 / (28. * 28.)).collect();
            images_as_v.push(v);
        }
    } // drop mnist_train_data
    // now read test data
    let mut image_fname = String::from(MNIST_FASHION_DIR);
    image_fname.push_str("t10k-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_FASHION_DIR);
    label_fname.push_str("t10k-labels-idx1-ubyte");
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    {
        let mnist_test_data  = MnistData::new(image_fname, label_fname).unwrap();
        let test_images = mnist_test_data.get_images();
        let mut test_labels = mnist_test_data.get_labels().to_vec();
        let( _, _, nbimages) = test_images.dim();
        let mut test_images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        //
        for k in 0..nbimages {
            let v : Vec<f32> = test_images.slice(s![.., .., k]).iter().map(|v| *v as f32 / (28.*28.)).collect();
            test_images_as_v.push(v);
        }
        labels.append(&mut test_labels);
        images_as_v.append(&mut test_images_as_v);
    } // drop mnist_test_data

    //
    // test mettu-plaxton or bmor algo
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    let distance = DistL2::default();
    match mnist_params.get_algo() {
        Algo::IMP   => {
            marrupaxton(&mnist_params, &images_as_v, &labels, distance)
        }
        Algo::BMOR   => {
            bmor(&mnist_params, &images_as_v, &labels, distance);
        }   
    }
    //
    let cpu_time: Duration = cpu_start.elapsed();
    println!("  sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_time.as_millis());
} // end of main


//============================================================================================



#[cfg(test)]

mod tests {


use super::*;

// test and compare some values obtained with Julia loading

#[test]
fn test_load_mnist_fashion() {
    let mut image_fname = String::from(MNIST_FASHION_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }

    let mut label_fname = String::from(MNIST_FASHION_DIR);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }

    let _mnist_data  = MnistData::new(image_fname, label_fname).unwrap();
    // check some value of the tenth images

} // end test_load


}  // end module tests