#!/bin/csh -f
#
#  Run GEOS implementation of NASA-AQcGAN to test running a forecast
#

source  /usr/share/modules/init/csh                                        
module purge

setenv HOME_DIR /discover/nobackup/pcastell/workspace/GEOSAQcGAN_update/GEOSAQcGAN
setenv PYTHONPATH ${HOME_DIR}/install/lib/Python
set BIN = ${HOME_DIR}/install/bin/
source $HOME_DIR/env@/g5_modules

# Set current directory
set cur_dir = "${PWD}"

# --> STEP 1: Preprocess Data [TBD]

set exp_name = test_one_mem

cp ${HOME_DIR}/src/tests/validate/norm_stats.pkl .

# for now link the preprocessed data to data_dir
set data_dir = "./data/geos_cf/${exp_name}"
set NORM_STATS_FILENAME="${data_dir}/norm_stats.pkl"                        
set geos_cf_yaml_fname="${HOME_DIR}/install/etc/NASA_AQcGAN/configs/geos_cf_preproc_collections.yaml"  

mkdir -p ${data_dir}
if ( ! -f ${data_dir}/norm_stats.pkl ) then                               
   ln -s ${cur_dir}/norm_stats.pkl ${data_dir}                                
endif                                                                      
                                                                           
python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf $NORM_STATS_FILENAME $data_dir $geos_cf_yaml_fname

# Symbolic links do not work yet when reading files. We need to copy for now.
cp ${data_dir}/*_meta.pkl ${data_dir}/meta.pkl
cp ${data_dir}/val/*_fields.npy ${data_dir}/val/1.npy
cp ${data_dir}/val/*_time.npy ${data_dir}/val/1_time.npy

#ln -s ${data_dir}/*_meta.pkl ${data_dir}/meta.pkl
#ln -s ${data_dir}/val/*_fields.npy ${data_dir}/val/1.npy
#ln -s ${data_dir}/val/*_time.npy ${data_dir}/val/1_time.npy

#ln -s $NOBACKUP/workspace/GEOSAQcGAN/GEOSAQcGAN/install/bin/tests/validate/data/geos_cf/test_one_mem/val ${data_dir}
#ln -s ${PWD}/norm_stats.pkl ${data_dir}
#ln -s $NOBACKUP/workspace/GEOSAQcGAN/GEOSAQcGAN/install/bin/tests/validate/data/geos_cf/test_one_mem/meta.pkl ${data_dir}

# --> STEP 2: Run the NASA-AQcGAN Model in inference mode

# link over some pickle and checkpoint files to exp_dir
set exp_dir = "./exp/${exp_name}"
mkdir -p ${exp_dir}

cp ${HOME_DIR}/src/tests/forecast/train_metrics.pkl .
cp ${HOME_DIR}/src/tests/forecast/val_metrics.pkl .

if ( ! -f ${exp_dir}/train_metrics.pkl ) then                               
   ln -s ${cur_dir}/train_metrics.pkl ${exp_dir}                                
endif                                                                      

if ( ! -f ${exp_dir}/val_metrics.pkl ) then                               
   ln -s ${cur_dir}/val_metrics.pkl ${exp_dir}                                
endif                                                                      

set MODEL_DIR=/discover/nobackup/projects/gmao/aist-cf/merged_aqcgan_inputs/checkpoint/projects/NOAA/climate-fast/ribaucj1/exp/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9/150
ln -s ${MODEL_DIR} ${exp_dir}

# input arguments
set META_FILEPATH=${data_dir}/meta.pkl
set CONFIG_FILEPATH=geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml
set CHKPT_IDX=150
set VERTICAL_LEVEL=72
set N_PASSES=1

python3 -m NASA_AQcGAN.inference.create_predictions $CONFIG_FILEPATH $CHKPT_IDX $META_FILEPATH --n_passes $N_PASSES --vertical_level $VERTICAL_LEVEL

# This final step creates a file in exp/test_one_mem called fc_pred_stats_days1_level72.npz
# This contains one array that has shape [32, 4, 8, 181, 360]

# 32 - number of time series predicted.  
# 4 - number of features predicted (CO,  NO, NO2, O3)
# 8 - number of frames (time steps) predicted
# 181, 360 - lat, lon

