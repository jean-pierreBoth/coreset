//! This module gets a DataMap from hnsw dump

use log;

use anyhow;

use hnsw_rs::datamap::*;

/// reloads a datamap and checks for type T.
pub fn get_typed_datamap<T: 'static + std::fmt::Debug>(
    directory: String,
    basename: String,
) -> anyhow::Result<DataMap> {
    //
    let res = DataMap::from_hnswdump::<T>(&directory, &basename);
    if res.is_err() {
        log::error!(
            "get_datamap, could not get datamap from hnsw, directory {}, basename : {}",
            directory,
            basename
        );
    }
    let datamap = res.unwrap();
    let t_name = datamap.get_data_typename();
    // check type
    let check_type = datamap.check_data_type::<T>();
    if !check_type {
        log::error!(
            "bad type name. registered type name : {}, you asked for {}",
            t_name,
            std::any::type_name::<T>().to_string()
        )
    }
    //
    return Ok(datamap);
}

//
pub fn get_datamap(directory: String, basename: String, typename: &str) -> anyhow::Result<DataMap> {
    //
    let _datamap = match &typename {
        &"u32" => get_typed_datamap::<u32>(directory, basename),
        &"u64" => get_typed_datamap::<u64>(directory, basename),
        _ => {
            log::error!("get_datamap : unimplemented type");
            std::panic!("get_datamap : unimplemented type");
        }
    };
    std::panic!("not yet");
}
