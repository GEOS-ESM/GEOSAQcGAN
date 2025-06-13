#!/bin/csh -f

#######################################################################
#                     Batch Parameters for Run Job
#######################################################################
 
#SBATCH -J geosaqcgan_fct
#SBATCH --gpus=1
#SBATCH --time=01:00:00
##SBATCH -A @GROUPID
#SBATCH -o output_geosaqcgan-%j.log
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

module load anaconda
conda activate torch

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

set cur_dir = "${PWD}"

set data_dir = "${cur_dir}/data"
mkdir -p ${data_dir}
#
set exp_dir = "${cur_dir}/exp"
mkdir -p ${exp_dir}

#######################################################################
#                   Set Experiment Run Variables
#######################################################################

# AQcGAN model directory
set MODEL_ROOT=/explore/nobackup/people/pcastell/aist-cf/nasa_cgan_model_march2025/
set CHKPT_IDX=150
set MODEL_DIR="${MODEL_ROOT}/projects/NOAA/climate-fast/ribaucj1/exp/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9/${CHKPT_IDX}"

#######################################################################
#                   STEP 1: Preprocess Data
#######################################################################
set NORM_STATS_FILENAME="${data_dir}/norm_stats.pkl"                        
set geos_cf_yaml_fname="${cur_dir}/geos_cf_preproc_collections.yaml"  
if ( $PREPROCESS_DATA == 1) then

# copy over model norm stats file
# it will be edited by the preprocess script
cp ${MODEL_ROOT}/norm_stats.pkl ${data_dir}                                

# run preprocess script                                                 
    echo "python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf \
        --norm_stats_file $NORM_STATS_FILENAME \
        --exp_dir $data_dir \
        --geos_cf_yaml_file $geos_cf_yaml_fname"
    
    python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf \
        --norm_stats_file $NORM_STATS_FILENAME \
        --exp_dir $data_dir \
        --geos_cf_yaml_file $geos_cf_yaml_fname
endif

#######################################################################
#          STEP 2: Run the NASA-AQcGAN Model in inference mode 
#######################################################################
# link over AQcGAN model files (pickle and checkpoint files) to exp_dir
if ( ! -f ${exp_dir}/train_metrics.pkl ) then
   ln -s ${MODEL_ROOT}/train_metrics.pkl ${exp_dir}
endif
if ( ! -f ${exp_dir}/val_metrics.pkl ) then
   ln -s ${MODEL_ROOT}/val_metrics.pkl ${exp_dir}
endif

if ( ! -d ${exp_dir}/${MODEL_DIR:t} ) then
    ln -s ${MODEL_DIR} ${exp_dir}
endif

# input arguments
set META_FILEPATH=${data_dir}/meta.pkl
set CONFIG_FILEPATH="${cur_dir}/config/forecast/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml"
set VERTICAL_LEVEL=72

set n_passes=1

set exp_name = `awk '/exp_name:/ {print $2}' $geos_cf_yaml_fname`
set beg_date = `awk '/beg_date:/ {print $2}' $geos_cf_yaml_fname`
set end_date = `awk '/end_date:/ {print $2}' $geos_cf_yaml_fname`

# Symbolic links to names the code expects.  Legacy - needs to be refactored.
ln -sf ${data_dir}/val/${exp_name}.${beg_date}-${end_date}.fields.npy ${data_dir}/val/1.npy
ln -sf ${data_dir}/val/${exp_name}.${beg_date}-${end_date}.time.npy ${data_dir}/val/1_time.npy
ln -sf ${data_dir}/${exp_name}.${beg_date}-${end_date}.meta.pkl ${data_dir}/meta.pkl

if ( ! -e ${data_dir}/val/1.npy || \
     ! -e ${data_dir}/val/1_time.npy || \
     ! -e ${data_dir}/meta.pkl ) then
    echo "Broken symbolic links for either .npy files or meta.pkl. Did you preprocess GEOS-CF data?"
    exit
endif

if ( $CLEAN_PREV_OUTPUT == 1 ) then
    rm -f ${exp_dir}/*aqcgan_predictions*.nc4
endif

while ( $n_passes <= $MAX_N_PASSES )
    echo "python3 -m NASA_AQcGAN.inference.create_predictions \
        --exp_dir ${exp_dir} \
        --config_filepath $CONFIG_FILEPATH \
        --chkpt_idx $CHKPT_IDX \
        --meta_filepath $META_FILEPATH \
        --n_passes $n_passes \
        --vertical_level $VERTICAL_LEVEL \
        --mode fcst"

    python3 -m NASA_AQcGAN.inference.create_predictions \
        --exp_dir ${exp_dir} \
        --config_filepath $CONFIG_FILEPATH \
        --chkpt_idx $CHKPT_IDX \
        --meta_filepath $META_FILEPATH \
        --n_passes $n_passes \
        --vertical_level $VERTICAL_LEVEL \
        --mode "fcst"
        
    if ($? != 0) then
        echo "Error running the model! Exiting..."
        exit(1)
    else
        @ n_passes++
    endif
end

# This final step creates a file in exp called 
# $expname.aqcgan_prediction.$startdate.nc4
# This contains 4 variables (CO,  NO, NO2, O3) that have shape [ntime, 181, 360]
# ntime - number of time series predicted.  
# 181, 360 - lat, lon
