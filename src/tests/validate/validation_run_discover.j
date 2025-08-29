#!/bin/csh -f

#######################################################################
#                     Batch Parameters for Run Job
#######################################################################
 
#SBATCH -J geosaqcgan_fct
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH -A @GROUPID
#SBATCH -o output_geosaqcgan_val-%j.log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
#######################################################################
#  Run GEOS implementation of NASA-AQcGAN to test running a forecast
#######################################################################
#           Architecture Specific Environment Variables
#######################################################################

setenv SRC_DIR @SRCDIR
setenv PYTHONPATH ${SRC_DIR}/install/lib/Python

source $SRC_DIR/env@/g5_modules

# This switch toggles whether we preprocess the GEOS-CF data 
# or use existing data
set PREPROCESS_DATA = 1

# This will delete all previous output
set CLEAN_PREV_OUTPUT = 0

# How many passes of the model to be run?
# Make sure that the time period (end - start) in the 
# geos_cf_yaml file is at least max_n_passes + 21 hrs
set MAX_N_PASSES=1

#######################################################################
#                 Create Experiment Sub-Directories
#######################################################################

set CUR_DIR = "${PWD}"

set DATA_DIR = "${CUR_DIR}/data"
mkdir -p ${DATA_DIR}
#
set EXP_DIR = "${CUR_DIR}/exp"
mkdir -p ${EXP_DIR}

#######################################################################
#                   Set Experiment Run Variables
#######################################################################

# AQcGAN model directory
set MODEL_ROOT=/discover/nobackup/projects/gmao/aist-cf/nasa_cgan_model_aug2025_v2.1.1/
set CHKPT_IDX=150
set MODEL_DIR="${MODEL_ROOT}/${CHKPT_IDX}"

#######################################################################
#                   STEP 1: Preprocess Data
#######################################################################
set NORM_STATS_FILENAME="${DATA_DIR}/norm_stats.pkl"                        
set geos_cf_yaml_fname="${CUR_DIR}/geos_cf_preproc_collections.yaml"  
if ( $PREPROCESS_DATA == 1) then

    # copy over model norm stats file
    # it will be edited by the preprocess script
    cp ${MODEL_ROOT}/norm_stats.pkl ${DATA_DIR}                                

    # run preprocess script                                                 
    set PRE_ARGS = "--norm_stats_file $NORM_STATS_FILENAME --exp_dir $DATA_DIR --geos_cf_yaml_file $geos_cf_yaml_fname --validation_file"

    echo "python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf $PRE_ARGS"
                                                                  
    python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf $PRE_ARGS

endif

#######################################################################
#          STEP 2: Run the NASA-AQcGAN Model in inference mode 
#######################################################################
# link over AQcGAN model files (pickle and checkpoint files) to exp_dir
if ( ! -f ${EXP_DIR}/train_metrics.pkl ) then
   ln -s ${MODEL_ROOT}/train_metrics.pkl ${EXP_DIR}
endif
if ( ! -f ${EXP_DIR}/val_metrics.pkl ) then
   ln -s ${MODEL_ROOT}/val_metrics.pkl ${EXP_DIR}
endif

#if ( ! -d ${EXP_DIR}/${MODEL_DIR:t} ) then
if ( ! -d ${EXP_DIR}/${MODEL_DIR} ) then
    ln -s ${MODEL_DIR} ${EXP_DIR}
endif

# input arguments
set META_FILEPATH=${DATA_DIR}/meta.pkl
set CONFIG_FILEPATH="${CUR_DIR}/config/validate/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml"
set VERTICAL_LEVEL=72

set n_passes=1

set exp_name = `awk '/exp_name:/ {print $2}' $geos_cf_yaml_fname`
set beg_date = `awk '/beg_date:/ {print $2}' $geos_cf_yaml_fname`
set end_date = `awk '/end_date:/ {print $2}' $geos_cf_yaml_fname`

# Symbolic links to names the code expects.  Legacy - needs to be refactored.
ln -sf ${DATA_DIR}/val/${exp_name}.${beg_date}-${end_date}.fields.npy ${DATA_DIR}/val/1.npy
ln -sf ${DATA_DIR}/val/${exp_name}.${beg_date}-${end_date}.time.npy ${DATA_DIR}/val/1_time.npy
ln -sf ${DATA_DIR}/${exp_name}.${beg_date}-${end_date}.meta.pkl ${DATA_DIR}/meta.pkl

mkdir -p ${DATA_DIR}/train                        
mkdir -p ${DATA_DIR}/test                         
ln -sf ${DATA_DIR}/val/${exp_name}.${beg_date}-${end_date}.fields.npy ${DATA_DIR}/train/1.npy
ln -sf ${DATA_DIR}/val/${exp_name}.${beg_date}-${end_date}.time.npy ${DATA_DIR}/train/1_time.npy
ln -sf ${DATA_DIR}/val/${exp_name}.${beg_date}-${end_date}.fields.npy ${DATA_DIR}/test/1.npy
ln -sf ${DATA_DIR}/val/${exp_name}.${beg_date}-${end_date}.time.npy ${DATA_DIR}/test/1_time.npy


if ( ! -e ${DATA_DIR}/val/1.npy || \
     ! -e ${DATA_DIR}/val/1_time.npy || \
     ! -e ${DATA_DIR}/meta.pkl ) then
    echo "Broken symbolic links for either .npy files or meta.pkl. Did you preprocess GEOS-CF data?"
    exit
endif

if ( $CLEAN_PREV_OUTPUT == 1 ) then
    rm -f ${EXP_DIR}/aqcgan_predictions/*.nc4
endif

while ( $n_passes <= $MAX_N_PASSES )
    set SPLIT  = "val"                           
    set VAL_ARGS = "$CONFIG_FILEPATH  $CHKPT_IDX ${DATA_DIR} --split $SPLIT --n_passes $n_passes --vertical_level $VERTICAL_LEVEL --is_pred"
              
    echo "python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS"
              
    python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS

    if ($? != 0) then
        echo "Error running the model! Exiting..."
        exit(1)
    else
        @ n_passes++
    endif
end

# This final step creates a file in exp/aqcgan_predictions called 
# $expname.aqcgan_prediction.$fcstdate.nc4
# This contains 4 variables (CO,  NO, NO2, O3) that have shape [ntime, 181, 360]
# If the file exists then it is appended with new data, but not overwritten.
# ntime - number of time series predicted.  
#         depends on the time interval of data provided to the model.  
#         will be number of input timesteps minus 8 * n_passes
# 181, 360 - lat, lon
