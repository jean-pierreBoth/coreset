//! Structure and functions to read MNIST fashion database
//! To run the examples change the line :  
//!
//! const MNIST_FASHION_DIR : &'static str = "/home.1/jpboth/Data/Fashion-MNIST/";
//!
//! command : mnist_fashion  --algo imp, bmor or coreset1
//! The coreset1 runs also a final kmedoids
//!
//! The data can be downloaded in the same format as the FASHION database from:  
//!
//! <https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion>
//!

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use anndists::dist::*;
use coreset::prelude::*;
use std::iter::Iterator;

use mnist::{check::*, io::*};

//============================================================================================

fn marrupaxton<Dist: Distance<f32> + Sync + Send + Clone>(
    _params: &MnistParams,
    images: &[Vec<f32>],
    labels: &[u8],
    distance: Dist,
) {
    //
    let mpalgo = MettuPlaxton::<f32, Dist>::new(images, distance);
    let alfa = 0.75;
    let mut facilities = mpalgo.construct_centers(alfa);
    //
    let (entropies, labels_distribution) = facilities.dispatch_labels(images, labels, None);
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
    mpalgo.compute_distances(&facilities);
}

//========================================================

fn bmor<Dist: Distance<f32> + Sync + Send + Clone>(
    _params: &MnistParams,
    images: &[Vec<f32>],
    labels: &[u8],
    distance: Dist,
) {
    //
    // if gamma increases, number of facilities increases.
    // if beta increases , upper bound on cost increases faster so the number of phases decreases
    let beta = 2.;
    let gamma = 2.;
    let mut bmor_algo: Bmor<usize, f32, Dist> = Bmor::new(10, 70000, beta, gamma, distance);
    //
    let ids = (0..images.len()).collect::<Vec<usize>>();
    let res = bmor_algo.process_data(images, &ids);
    if res.is_err() {
        std::panic!("bmor failed");
    }
    //
    // do we ask for a supplementary contraction pass
    let contraction = false;
    let mut facilities = bmor_algo.end_data(contraction);
    //
    let (entropies, labels_distribution) = facilities.dispatch_labels(images, labels, None);
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

//=====================================================================

pub fn parse_cmd(matches: &ArgMatches) -> Result<MnistParams, anyhow::Error> {
    log::debug!("in parse_cmd");
    if matches.contains_id("algo") {
        println!("decoding argument algo");
        let algoname = matches.get_one::<String>("algo").expect("");
        log::debug!(" got algo : {:?}", algoname);
        match algoname.as_str() {
            "imp" => {
                let params = MnistParams::new(Algo::IMP);
                return Ok(params);
            }
            "bmor" => {
                let params = MnistParams::new(Algo::BMOR);
                return Ok(params);
            }
            "coreset1" => {
                let params = MnistParams::new(Algo::CORESET1);
                return Ok(params);
            }
            //
            _ => {
                log::error!(" algo must be imp or bmor or coreset1 ");
                std::process::exit(1);
            }
        }
    }
    //
    Err(anyhow::anyhow!("bad command"))
} // end of parse_cmd

//========================================================

use clap::{Arg, ArgAction, ArgMatches, Command};

const MNIST_FASHION_DIR_NOT_CSV: &str = "/home/jpboth/Data/ANN/Fashion-MNIST/";
const MNIST_FASHION_DIR_CSV: &str = "/home/jpboth/Data/MnistFashionCsv";

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    log::info!("\n\n running mnist_fashion \n ==========================");
    //
    let matches = Command::new("mnist_fashion")
        //        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            Arg::new("algo")
                .required(true)
                .long("algo")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting a algo option imp, bmor or coreset1"),
        )
        .get_matches();
    //
    let mnist_params = parse_cmd(&matches).unwrap();
    //
    //
    let csv_format = false;
    //
    let (labels, images_as_v) = if csv_format {
        log::info!(
            "in mnist_digits, reading mnist data original idx bianry ...from {}",
            MNIST_FASHION_DIR_CSV
        );
        io_from_csv(MNIST_FASHION_DIR_CSV).unwrap()
    } else {
        log::info!(
            "in mnist_digits, reading mnist data in CSV format ...from {}",
            MNIST_FASHION_DIR_NOT_CSV
        );
        io_from_non_csv(MNIST_FASHION_DIR_NOT_CSV).unwrap()
    };
    //
    // test mettu-plaxton or bmor algo
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    let distance = DistL1;
    match mnist_params.get_algo() {
        Algo::IMP => marrupaxton(&mnist_params, &images_as_v, &labels, distance),
        Algo::BMOR => {
            bmor(&mnist_params, &images_as_v, &labels, distance);
        }
        Algo::CORESET1 => {
            coreset1(&mnist_params, &images_as_v, &labels, distance);
        }
    }
    //
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        "  sys time(ms) {:?} cpu time(ms) {:?}",
        sys_now.elapsed().unwrap().as_millis(),
        cpu_time.as_millis()
    );
} // end of main

//============================================================================================

#[cfg(test)]

mod tests {

    use super::*;
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    // test and compare some values obtained with Julia loading

    #[test]
    fn test_load_mnist_fashion() {
        let mut image_fname = String::from(MNIST_FASHION_DIR_NOT_CSV);
        image_fname.push_str("train-images-idx3-ubyte");
        let image_path = PathBuf::from(image_fname.clone());
        let image_file_res = OpenOptions::new().read(true).open(&image_path);
        if image_file_res.is_err() {
            println!("could not open image file : {:?}", image_fname);
            return;
        }

        let mut label_fname = String::from(MNIST_FASHION_DIR_NOT_CSV);
        label_fname.push_str("train-labels-idx1-ubyte");
        let label_path = PathBuf::from(label_fname.clone());
        let label_file_res = OpenOptions::new().read(true).open(&label_path);
        if label_file_res.is_err() {
            println!("could not open label file : {:?}", label_fname);
            return;
        }

        let _mnist_data = MnistData::new(image_path, label_path).unwrap();
        // check some value of the tenth images
    } // end test_load
} // end module tests
