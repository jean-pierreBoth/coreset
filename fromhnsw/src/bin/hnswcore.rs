//! This binary is dedicated to coreset computations on data stored in Hnsw created by crate [hnsw_rs](https://crates.io/crates/hnsw_rs)
//!
//! command is :hnscore  --dir (-d) dirname  --fname (-f) hnswname  --typename (-t) typename [--beta b] [--gamma g]
//!
//! - dirname : directory where hnsw files reside
//! - hnswname : name used for naming the 2 hnsw related files: name.hnsw.data and name.hnsw.graph
//! - typename : can be u16, u32, u64, f32, f64, i16, i32, i64
//!
//! The coreset command takes as arguments:
//! - beta:
//! - gamma:
//!
//! Note: It is easy to add any adhoc type T  by adding a line in [get_datamap()].  
//! The only constraints on T comes from hnsw and is T: 'static + Clone + Sized + Send + Sync + std::fmt::Debug

//#![allow(unused)]

use std::fs::OpenOptions;
use std::path::PathBuf;

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use coreset::prelude::*;

use clap::{Arg, ArgAction, ArgMatches, Command};
use std::default::Default;

use anndists::dist::*;
use fromhnsw::getdatamap::get_typed_datamap;
use hnsw_rs::datamap::*;

use fromhnsw::hnswiter::HnswMakeIter;

//========================================
// Parameters

#[derive(Debug, Clone)]
struct HcoreParams {
    path: HnswParams,
    corearg: CoresetParams,
}
#[derive(Debug, Clone)]
struct HnswParams {
    dir: String,
    hname: String,
    typename: String,
}

impl HnswParams {
    pub fn new(hdir: &String, hname: &String, typename: &String) -> Self {
        HnswParams {
            dir: hdir.clone(),
            hname: hname.clone(),
            typename: typename.clone(),
        }
    }
}

//
/// Coreset parameters
#[derive(Copy, Clone, Debug)]
pub struct CoresetParams {
    beta: f32,
    gamma: f32,
}

impl CoresetParams {
    fn new(beta: f32, gamma: f32) -> CoresetParams {
        CoresetParams { beta, gamma }
    }
    //
    fn get_beta(&self) -> f32 {
        self.beta
    }

    //
    fn get_gamma(&self) -> f32 {
        self.gamma
    }
}

impl Default for CoresetParams {
    fn default() -> Self {
        CoresetParams {
            beta: 2.,
            gamma: 2.,
        }
    }
}

#[derive(Clone, Debug)]
struct HnswCore {
    // paths
    hparams: HnswParams,
    // algo parameters
    coreparams: CoresetParams,
}

//===========================================================

fn parse_coreset_cmd(matches: &ArgMatches) -> Result<CoresetParams, anyhow::Error> {
    log::debug!("in  parse_coreset_cmd");
    let mut params = CoresetParams::default();
    params.beta = *matches.get_one::<f32>("beta").unwrap();
    params.gamma = *matches.get_one::<f32>("gamma").unwrap();
    //
    log::info!("got CoresetParams : {:?}", params);
    //
    return Ok(params);
}

//============================================================================================

/// This function dispatch its call to get_typed_datamap::\<T\> according to type T
/// The cuurent function dispatch to u16, u32, u64, i32, i64, f32 and f64 according to typename.
/// For another type, the functio is easily modifiable.  
/// The only constraints on T comes from hnsw and is T: 'static + Clone + Sized + Send + Sync + std::fmt::Debug
pub fn get_datamap(directory: String, basename: String, typename: &str) -> anyhow::Result<DataMap> {
    //
    let _datamap = match &typename {
        &"u16" => get_typed_datamap::<u16>(directory, basename),
        &"u32" => get_typed_datamap::<u32>(directory, basename),
        &"u64" => get_typed_datamap::<u64>(directory, basename),
        &"f32" => get_typed_datamap::<f32>(directory, basename),
        &"f64" => get_typed_datamap::<f64>(directory, basename),
        &"i32" => get_typed_datamap::<i32>(directory, basename),
        &"i64" => get_typed_datamap::<i64>(directory, basename),
        _ => {
            log::error!(
                "get_datamap : unimplemented type, type received : {}",
                typename
            );
            std::panic!("get_datamap : unimplemented type");
        }
    };
    std::panic!("not yet");
}

//
/*
use anndists::dist::*;
macro_rules! implement_get_l1(
    ($ty:ty) => $ty{}
);

implement_get_l1!(DistL1);

*/

pub fn get_distance_l1() -> DistL1 {
    DistL1 {}
}

pub fn get_distance<T: Send + Sync>(distname: &str) -> Box<dyn Distance<T>>
where
    DistL1: Distance<T>,
    DistL2: Distance<T>,
    DistJaccard: Distance<T>,
{
    match distname {
        "DistL1" => Box::new(DistL1 {}),
        "DistL2" => Box::new(DistL2 {}),
        "DistJaccard" => Box::new(DistJaccard {}),
        _ => std::panic!("get_distance got bad distance"),
    }
    //  std::panic!("not yet");
}

//===========================================================

pub fn coreset1<T, Dist>(coreparams: &CoresetParams, datamap: &DataMap, distance: Dist)
where
    T: Send + Sync + Clone + std::fmt::Debug,
    Dist: Distance<T> + Sync + Send + Clone,
{
    //
    println!("\n\n entering coreset + our kmedoids");
    println!("==================================");
    //
    let beta = coreparams.get_beta().into();
    let gamma = coreparams.get_gamma().into();
    let nb_data = 50000;
    //
    let dl1 = DistL1 {};
    let k = 10; // as we have 10 classes, but this gives a lower bound
    let mut core1 = Coreset1::<usize, T, Dist>::new(k, nb_data, beta, gamma, distance.clone());
    //
    let iterproducer = HnswMakeIter::<T>::new(datamap);
    //

    let res = core1.make_coreset(&iterproducer, 0.11);
    if res.is_err() {
        log::error!("construction of coreset1 failed");
    }
    let coreset = res.unwrap();
    // get some info
    log::info!("coreset1 nb different points : {}", coreset.get_nb_points());
    //
    let dist_name = std::any::type_name::<Dist>();
    log::info!("dist name = {:?}", dist_name);
} // end of

//===========================================================

fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    log::info!("running hnswcore");
    //
    let hparams: HnswParams;
    let core_params: CoresetParams;

    //
    let params: CoresetParams;
    let coresetcmd = Command::new("coreset")
        .arg(
            Arg::new("beta")
                .required(false)
                .short('b')
                .long("beta")
                .default_value("2.0")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f32))
                .help("beta"),
        )
        .arg(
            Arg::new("gamma")
                .required(false)
                .short('g')
                .long("gamma")
                .default_value("2.0")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f32))
                .help("gamma"),
        );
    //
    // global command
    // =============
    //
    let matches = Command::new("hnswcore")
        .arg_required_else_help(true)
        .arg(
            Arg::new("dir")
                .long("dir")
                .short('d')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting a directory name"),
        )
        .arg(
            Arg::new("fname")
                .long("fname")
                .short('f')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting a file  basename"),
        )
        .arg(
            Arg::new("typename")
                .short('t')
                .long("type")
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting a directory name"),
        )
        .subcommand(coresetcmd)
        .get_matches();
    //
    // retrieve HnswPathParams
    //
    let hdir = matches
        .get_one::<String>("dir")
        .expect("dir argument needed");
    let hname = matches
        .get_one::<String>("fname")
        .expect("hnsw base name needed");
    let tname: &String = matches
        .get_one::<String>("fname")
        .expect("typename required");
    //
    let hparams = HnswParams::new(hdir, hname, tname);
    //
    // parse coreset parameters
    //
    if let Some(core_match) = matches.subcommand_matches("coreset") {
        log::debug!("subcommand for coreset parameters");
        let res = parse_coreset_cmd(core_match);
        match res {
            Ok(params) => {
                core_params = params;
            }
            _ => {
                log::error!("parsing coreset command failed");
                println!("exiting with error {}", res.err().as_ref().unwrap());
                //  log::error!("exiting with error {}", res.err().unwrap());
                std::process::exit(1);
            }
        }
    } else {
        core_params = CoresetParams::default();
    }
    log::debug!("coreset params : {:?}", core_params);
    // retrieve
    //
    // Datamap Creation
    //
    let typename = "u32";
    let datamap = get_datamap(hparams.dir, hparams.hname, typename);
    if datamap.is_err() {
        log::error!(
            "datamap could not be constructed : {}",
            datamap.err().unwrap()
        );
        std::process::exit(1);
    }
    let datamap = datamap.unwrap();
    // Distance instanciation
    let distname = datamap.get_distname();
    let distance = get_distance::<u32>(&distname);
}
