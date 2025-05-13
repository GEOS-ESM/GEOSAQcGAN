#!/bin/csh -f
#
#  Run GEOS implementation of NASA-AQcGAN to test running a forecast
#

setenv HOME_DIR /discover/nobackup/pcastell/workspace/GEOSAQcGAN_update/GEOSAQcGAN/ 
setenv PYTHONPATH ${HOME_DIR}/install/lib/Python
set BIN = ${HOME_DIR}/install/bin/
source $HOME_DIR/env@/g5_modules


# STEP 1: Preprocess Data [TBD]

set exp_name = test_one_mem

# for now link the preprocessed data to data_dir
set data_dir = ./data/geos_cf/${exp_name}
mkdir -p ${data_dir}
ln -s $NOBACKUP/workspace/GEOSAQcGAN/GEOSAQcGAN/install/bin/tests/validate/data/geos_cf/test_one_mem/val ${data_dir}
ln -s ${PWD}/norm_stats.pkl ${data_dir}
ln -s $NOBACKUP/workspace/GEOSAQcGAN/GEOSAQcGAN/install/bin/tests/validate/data/geos_cf/test_one_mem/meta.pkl ${data_dir}

# STEP 2: Run the NASA-AQcGAN Model in inference mode

# link over some pickle and checkpoint files to exp_dir
set exp_dir = ./exp/${exp_name}
mkdir -p ${exp_dir}
ln -s ${PWD}/train_metrics.pkl ${exp_dir}
ln -s ${PWD}/val_metrics.pkl   ${exp_dir}

set MODEL_DIR=/discover/nobackup/projects/gmao/aist-cf/merged_aqcgan_inputs/checkpoint/projects/NOAA/climate-fast/ribaucj1/exp/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9/150
ln -s ${MODEL_DIR} ${exp_dir}

# input arguments
set META_FILEPATH=${data_dir}/meta.pkl
set CONFIG_FILEPATH=geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml
set CHKPT_IDX=150
set VERTICAL_LEVEL=72
set N_PASSES=1

python3 -im NASA_AQcGAN.inference.create_predictions $CONFIG_FILEPATH $CHKPT_IDX $META_FILEPATH --n_passes $N_PASSES --vertical_level $VERTICAL_LEVEL

# This final step creates a file in exp/test_one_mem called fc_pred_stats_days1_level72.npz
# This contains one array that has shape [32, 4, 8, 181, 360]

# 32 - number of time series predicted.  
# 4 - number of features predicted (CO,  NO, NO2, O3)
# 8 - number of frames (time steps) predicted
# 181, 360 - lat, lon

