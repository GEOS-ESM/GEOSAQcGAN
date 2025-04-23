#!/bin/csh -f
#
#  Run NASA-AQcGAN in validatoin mode with test dataset of just one ensemble member
#

setenv HOME_DIR /discover/nobackup/pcastell/workspace/GEOSAQcGAN/GEOSAQcGAN/ 
setenv PYTHONPATH ${HOME_DIR}/src/AQcGAN@/src
set BIN = ${HOME_DIR}/install/bin/
source $HOME_DIR/env@/g5_modules


# STEP 1: Merge GEOS Datsets

set perturb=met_and_emis
set data_type=gcc
set member=1
set suffix=my_test
set root_dir=/discover/nobackup/projects/gmao/aist-cf/merged_aqcgan_inputs

python3 ${BIN}/NASA_AQcGAN/merge_geos_cf_ens_data.py $perturb $data_type $member -suffix $suffix --data_dir $root_dir

# $root_dir contains a GEOS-CF dataset of 40 time steps (5 days) copied over from 
# /css/gmao/geos-cf-dev/pub/AIST21_0024/TrainingData_Ensembles/c90_v0
# This step will create a file called gcc_my_test.pkl in $root_dir/mem001

# This pickle file has all of the variables concatenated into shapes [40,181,360] --> [ntime,nlat,nlon]
# Variables that have multiple levels are renamed to e.g. SpeciesConc_O3_34 --> $NAME_$lev 


# STEP 2: Normalize Merged Datasets
set exp_name=test_one_mem
set EXP_DIR=data/geos_cf/${exp_name}
set data_file=gcc_my_test
set member=1
set NORM_STATS_FILENAME="${EXP_DIR}/norm_stats.pkl"

mkdir -p ${EXP_DIR}
ln -s norm_stats.pkl ${EXP_DIR}

python3 -i ${BIN}/NASA_AQcGAN/preprocess_geos_cf_ens.py $perturb $data_type $data_file $NORM_STATS_FILENAME $member $EXP_DIR --root_dir $root_dir --train_members 1 --test_members=1 --val_members=1

# The code preprocess_geos_cf_ens.py reads in all the merged files from STEP 1 and normalizes the data.
# The script takes as input indices of the members that it should use for training, testing, and validation.
# The indices are in reference to the members located in the $root_dir
# Since I only have copied over 3 ensemble members into $root_dir, the indices can only be 1-3.
# I'm trying to replicate a case of just having one data stream, so have used index=1 for the train, test, and val datasets.

# This step creates directories inside $EXP_DIR called train, test, val.  
# Inside these directories are numpy files called $member_number.npy and $member_number_time.npy.  
# There's a decision tree inside preprocess_geos_cf_ens.py that makes it so only "train" and "val" 
# preprocessed data is made when the same member number is used for all three.


# STEP 3: Run the NASA-AQcGAN Model in inference mode

mkdir -p exp/${exp_name}
ln -s train_metrics.pkl exp/${exp_name}
ln -s val_metrics.pkl exp/${exp_name}

set MODEL_DIR=/discover/nobackup/projects/gmao/aist-cf/merged_aqcgan_inputs/checkpoint/projects/NOAA/climate-fast/ribaucj1/exp/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9/150
ln -s ${MODEL_DIR} exp/${exp_name}

set CONFIG_FILE=geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml
set CUDA_VISIBLE_DEVICES=0
set DATA_DIR=${exp_name}
set CHKPT_IDX=150
set VERTICAL_LEVEL=72
set SPLIT="val"
set N_PASSES=1

#python3 -im inference.create_ensemble_predictions $CONFIG_FILE $CHKPT_IDX $DATA_DIR --n_passes $N_PASSES --vertical_level $VERTICAL_LEVEL --split $SPLIT --is_pred

# This final step creates a file in exp/test_one_mem called val_ens_pred_stats_days1_level72.npz
# This contains two arrays - one for the mean, one for the standard deviation of the ensembles (in this case it's just one ensemble member, so the std is just all NAN). In this case it has shape [24, 4, 8, 181, 360]

# 24 - number of time series predicted.  This code doesn't make predictions for the last 2*n_frames*step_size = 16 time steps. 
# This is because it's set up to implicitly require that ground-truth (target) data be available for the predictions.

# 4 - number of features predicted (CO,  NO, NO2, O3)

# 8 - number of frames (time steps) predicted

# 181, 360 - lat, lon

