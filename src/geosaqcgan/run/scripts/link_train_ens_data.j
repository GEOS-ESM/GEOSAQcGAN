#!/bin/csh -fx
# Header
# Description: Script to process and link data files based on YAML configuration
# Usage: ./link_train_ens_data.j data_dir preprocess.config
# Author: Viral Shah
# Date: 2025-10-30
 
set aqcgan_config = $1
source $aqcgan_config

set preproc_config = $2
source $preproc_config

set src_dir = $GEOSdata_dir
set dest_dir = $DATA_DIR/$staging_dir/$perturb

# Delete everything from dest_dir
rm -rf $dest_dir/*

# Loop through timestamps
set date_part = `echo $start_timestamp | cut -c1-8`
set time_part = `echo $start_timestamp | cut -c10-13 | sed 's/\(..\)/\1:/'`
set start_time_s = `date -d "$date_part $time_part" "+%s"`

set date_part = `echo $end_timestamp | cut -c1-8`
set time_part = `echo $end_timestamp | cut -c10-13 | sed 's/\(..\)/\1:/'`
set end_time_s = `date -d "$date_part $time_part" "+%s"`

set current_time_s = $start_time_s
while ($current_time_s <= $end_time_s)
    set current_timestamp = `date -d "@$current_time_s" +%Y%m%d_%H%M`
    foreach mem ($members)
        foreach coll ($collections)
            set src_dir_mem = "${src_dir}/mem${mem}/$coll/"
            set dest_dir_mem = "${dest_dir}/mem${mem}/$coll/"

            # Create destination directory if it doesn't exist
            mkdir -p $dest_dir_mem
            
            # Construct the file name pattern
            set file_pattern = "GCC_c90_met_emis_mem${mem}.$coll.${current_timestamp}z.nc4"

            # Find matching files and create soft links
            find ${src_dir_mem} -name $file_pattern -type f -exec ln -s {} ${dest_dir_mem} \;
        end
    end

    # Increment timestamp by the specified interval
    set current_time_s = `expr $current_time_s + $time_increment \* 3600`

end
