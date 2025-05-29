//! Structure and functions to read MNIST digits database
//! To run the examples change in main the line :  
//!
//! const MNIST_DIGITS_DIR_CSV : &'static str = "/home/jpboth/Data/MNIST/";
//!
//! to whatever directory you downloaded the [MNIST digits data](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use anndists::dist::*;

use mnist::{check::*, io::*};

//============================================================================================

pub fn parse_cmd(matches: &ArgMatches) -> Result<MnistParams, anyhow::Error> {
    log::debug!("in parse_cmd");
    if matches.contains_id("algo") {
        log::debug!("decoding argument algo");
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
                log::error!(" algo must be imp or bmor");
                std::process::exit(1);
            }
        }
    }
    //
    Err(anyhow::anyhow!("bad command"))
} // end of parse_cmd

//=============================================================================================

fn marrupaxton<Dist: Distance<f32> + Sync + Send + Clone>(
    _params: &MnistParams,
    images: &[Vec<f32>],
    labels: &[u8],
    distance: Dist,
) {
    //
    log::info!("in marrupaxton");
    //
    let mpalgo = MettuPlaxton::<f32, Dist>::new(images, distance);
    let alfa = 1.;
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
} // end of marrupaxton

//=================================================================================================

fn bmor<Dist: Distance<f32> + Sync + Send + Clone>(
    _params: &MnistParams,
    images: &[Vec<f32>],
    labels: &[u8],
    distance: Dist,
) {
    //
    log::info!("in bmor");
    // we increase a little coefficients to get more facilities
    let beta = 2.2;
    let gamma = 2.2;
    let mut bmor_algo = Bmor::new(10, 70000, beta, gamma, distance);
    //
    let ids = (0..images.len()).collect::<Vec<usize>>();
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
} // end of bmor

//=====================================================================

use clap::{Arg, ArgAction, ArgMatches, Command};

use coreset::prelude::*;

// for data in old non csv format
const MNIST_DIGITS_DIR_NOT_CSV: &str = "/home/jpboth/Data/ANN/MNIST";

// for data in csv format
const MNIST_DIGITS_DIR_CSV: &str = "/home/jpboth/Data/MnistDigitsCsv";

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    log::info!("\n\n running mnist_digits");
    //
    let matches = Command::new("mnist_digits")
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
    let csv_format = false;
    //
    let (labels, images_as_v) = if csv_format {
        log::info!(
            "in mnist_digits, reading mnist data original idx bianry ...from {}",
            MNIST_DIGITS_DIR_CSV
        );
        io_from_csv(MNIST_DIGITS_DIR_CSV).unwrap()
    } else {
        log::info!(
            "in mnist_digits, reading mnist data in CSV format ...from {}",
            MNIST_DIGITS_DIR_NOT_CSV
        );
        io_from_non_csv(MNIST_DIGITS_DIR_NOT_CSV).unwrap()
    };
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
} // end of main digits

//============================================================================================
