#!/bin/csh -f

#######################################################################
#                     Batch Parameters for Run Job
#######################################################################
 
#SBATCH -J geosaqcgan_fct
#SBATCH --nodes=1
#SBATCH --constraint=mil
#SBATCH --time=01:00:00
#SBATCH -A @GROUPID
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

source $SRC_DIR/env@/g5_modules

#######################################################################
#                 Create Experiment Sub-Directories
#######################################################################

set cur_dir = "${PWD}"
set exp_name = test_one_mem

set data_dir = "./data/geos_cf/${exp_name}"
mkdir -p ${data_dir}
#
set exp_dir = "./exp/${exp_name}"
mkdir -p ${exp_dir}

#######################################################################
#                   Set Experiment Run Variables
#######################################################################

# AQcGAN model directory
set MODEL_ROOT=/discover/nobackup/projects/gmao/aist-cf/nasa_cgan_model_march2025/
set CHKPT_IDX=150
set MODEL_DIR="${MODEL_ROOT}/projects/NOAA/climate-fast/ribaucj1/exp/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9/${CHKPT_IDX}"
#set MODEL_DIR=/discover/nobackup/projects/gmao/aist-cf/merged_aqcgan_inputs/checkpoint/projects/NOAA/climate-fast/ribaucj1/exp/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9/150

#######################################################################
#                   STEP 1: Preprocess Data
#######################################################################

set NORM_STATS_FILENAME="${data_dir}/norm_stats.pkl"                        
set geos_cf_yaml_fname="${SRC_DIR}/install/etc/NASA_AQcGAN/configs/geos_cf_preproc_collections.yaml"  

# copy over model norm stats file
# it will be edited by the preprocess script
cp ${MODEL_ROOT}/norm_stats.pkl ${data_dir}                                
                                                                           
python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf $NORM_STATS_FILENAME $data_dir $geos_cf_yaml_fname

# Symbolic links to names the code expects.  Legacy - needs to be refactored.
if ( ! -f ${cur_dir}/${data_dir}/val/1.npy ) then
    ln -s ${cur_dir}/${data_dir}/val/*_fields.npy ${cur_dir}/${data_dir}/val/1.npy
endif
if ( ! -f ${cur_dir}/${data_dir}/val/1_time.npy ) then
    ln -s ${cur_dir}/${data_dir}/val/*_time.npy ${cur_dir}/${data_dir}/val/1_time.npy
endif
if ( ! -f ${cur_dir}/${data_dir}/meta.pkl ) then
    ln -s ${cur_dir}/${data_dir}/*_meta.pkl ${cur_dir}/${data_dir}/meta.pkl
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
set CONFIG_FILEPATH="${SRC_DIR}/install/bin/tests/validate/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml"
set VERTICAL_LEVEL=72
set N_PASSES=1

python3 -m NASA_AQcGAN.inference.create_predictions $CONFIG_FILEPATH $CHKPT_IDX $META_FILEPATH --n_passes $N_PASSES --vertical_level $VERTICAL_LEVEL

# This final step creates a file in exp/test_one_mem called fc_pred_stats_days1_level72.npz
# This contains one array that has shape [ntime, 4, 8, 181, 360]

# ntime - number of time series predicted.  depends on the size interval of data provided to the model
# 4 - number of features predicted (CO,  NO, NO2, O3)
# 8 - number of frames (time steps) predicted
# 181, 360 - lat, lon

