//! Structure and functions to read MNIST digits database
//! To run the examples change the line :  
//! 
//! const MNIST_DIGITS_DIR : &'static str = "/home/jpboth/Data/MNIST/";
//! 
//! to whatever directory you downloaded the [MNIST digits data](http://yann.lecun.com/exdb/mnist/)

use ndarray::s;
use std::fs::OpenOptions;
use std::path::PathBuf;

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;


use hnsw_rs::prelude::*;


mod utils;
use utils::{mnistio::*, mnistcheck::*};

//============================================================================================



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
            "coreset1" => {
                let params = MnistParams::new(Algo::CORESET1);
                return Ok(params);
            }  
            //
            _          => {
                log::error!(" algo must be imp or bmor");
                std::process::exit(1);
            }
        }
    }
    //
    return Err(anyhow::anyhow!("bad command"));
} // end of parse_cmd



//=============================================================================================

fn marrupaxton<Dist : Distance<f32> + Sync + Send + Clone>(_params :&MnistParams, images : &Vec<Vec<f32>>, labels : &Vec<u8>, distance : Dist) {
    //
    log::info!("in marrupaxton");
    //
    let mpalgo = MettuPlaxton::<f32, Dist>::new(&images, distance);
    let alfa = 1.;
    let mut facilities = mpalgo.construct_centers(alfa);
    //
    let (entropies, labels_distribution) = facilities.dispatch_labels(&images , &labels, None);
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
    mpalgo.compute_distances(&mut facilities);
}  // end of marrupaxton

//=================================================================================================

fn bmor<Dist : Distance<f32> + Sync + Send + Clone>(_params :&MnistParams, images : &Vec<Vec<f32>>, labels : &Vec<u8>, distance : Dist) {
    //
    log::info!("in bmor");
    // we increase a little coefficients to get more facilities
    let beta = 2.2;
    let gamma = 2.2;
    let mut bmor_algo = Bmor::new(10, 70000, beta, gamma, distance);
    //
    let ids = (0..images.len()).into_iter().collect::<Vec<usize>>();
    let res = bmor_algo.process_data(images, &ids);
    if res.is_err() {
        std::panic!("bmor failed");
    }
    let nb_facility = res.unwrap();
    log::info!("got nb facilities : {:?}", nb_facility);
    // do we ask for a supplementary contraction pass
    let contraction = false;
    //============================
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
} // end of bmor

//=====================================================================


use clap::{Arg, ArgMatches, ArgAction, Command};


use coreset::prelude::*;

const MNIST_DIGITS_DIR : &'static str = "/home/jpboth/Data/ANN/MNIST/";

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
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DIGITS_DIR);
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
            let v : Vec<f32> = images.slice(s![.., .., k]).iter().map(|v| *v as f32 / (28. * 28.)).collect();
            images_as_v.push(v);
        }
    } // drop mnist_train_data
    // now read test data
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("t10k-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DIGITS_DIR);
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
            let v : Vec<f32> = test_images.slice(s![.., .., k]).iter().map(|v| *v as f32 / (28. * 28.)).collect();
            test_images_as_v.push(v);
        }
        labels.append(&mut test_labels);
        images_as_v.append(&mut test_images_as_v);
    } // drop mnist_test_data
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    let distance = DistL1::default();
    match mnist_params.get_algo() {
        Algo::IMP   => {
            marrupaxton(&mnist_params, &images_as_v, &labels, distance)
        }
        Algo::BMOR   => {
            bmor(&mnist_params, &images_as_v, &labels, distance);
        }
        Algo::CORESET1 => {
            coreset1(&mnist_params, &images_as_v, &labels, distance);
        } 
    }
    //
    let cpu_time: Duration = cpu_start.elapsed();
    println!("  sys time(ms) {:?} cpu time(ms) {:?}", sys_now.elapsed().unwrap().as_millis(), cpu_time.as_millis());
}  // end of main digits


//============================================================================================



#[cfg(test)]

mod tests {


use super::*;

// test and compare some values obtained with Julia loading

#[test]

fn test_load_mnist() {
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }

    let mut label_fname = String::from(MNIST_DIGITS_DIR);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }

    let mnist_data  = MnistData::new(image_fname, label_fname).unwrap();
    assert_eq!(0x3c, *mnist_data.images.get([9,14,9]).unwrap());
    assert_eq!(0xfd, mnist_data.images[(14 , 9, 9)]);
    // check some value of the tenth images

    // check first and last labels
    assert_eq!(5, mnist_data.labels[0]);
    assert_eq!(8, mnist_data.labels[mnist_data.labels.len()-1]);
    assert_eq!(1,1);
} // end test_load


}  // end module tests