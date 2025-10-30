    #!/bin/csh -f

#######################################################################
#                     Batch Parameters for Run Job
#######################################################################
 
#SBATCH -J geosaqcgan_fct
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH -A s2825
#SBATCH -o output_geosaqcgan_val-%j.log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
#######################################################################
#  Run GEOS implementation of NASA-AQcGAN to test running a forecast
#######################################################################
#           Architecture Specific Environment Variables
#######################################################################

setenv SRC_DIR /gpfsm/dnb06/projects/p271/vshah5/GEOSAQcGAN
setenv PYTHONPATH ${SRC_DIR}/install/lib/Python

source $SRC_DIR/env@/g5_modules

# This switch toggles whether we preprocess the GEOS-CF data 
# or use existing data
set PREPROCESS_DATA = 0

# This will delete all previous output
set CLEAN_PREV_OUTPUT = 0

# How many passes of the model to be run?
# Make sure that the time period (end - start) in the 
# geos_cf_yaml file is at least max_n_passes + 21 hrs
set MAX_N_PASSES = 4

set EXP_NAME = "TrainEns"

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

set CONFIG_FILEPATH="${CUR_DIR}/config/validate/geos_cf_perturb_met_and_emis_gcc_feb_sep_surface_only_time_8ts_nolstm_nolatlon_none_train_7_28_12_17_29_3_1_25_20_19_24_23_22_15_8_26_21_5_9.yaml"

set VERTICAL_LEVEL=72

set SPLIT  = "test"

#######################################################################
#                   STEP 1: Preprocess Data
#######################################################################
set NORM_STATS_FILENAME="${DATA_DIR}/norm_stats.pkl"                        
set geos_cf_yaml_fname="${CUR_DIR}/geos_cf_preproc_collections.yaml"  
if ( $PREPROCESS_DATA == 1) then
    # link GEOS data
    ./link_training_ensemble_data.j

    # preprocess
    ./preprocess_geos_ens_data.sh

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

# copy model norm stats file
# it will be edited by the preprocess script
cp ${MODEL_ROOT}/norm_stats.pkl ${DATA_DIR}

set n_passes=1

if ( $CLEAN_PREV_OUTPUT == 1 ) then
    rm -f ${EXP_DIR}/aqcgan_predictions/*.nc4
    rm -f ${EXP_DIR}/*stats*.npz
endif

while ( $n_passes <= $MAX_N_PASSES )
    # Get predictions first
    set VAL_ARGS = "$CONFIG_FILEPATH  $CHKPT_IDX ${DATA_DIR} --split $SPLIT --n_passes $n_passes --vertical_level $VERTICAL_LEVEL --is_pred"

    echo "python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS"

#    python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS

    if ($? != 0) then
        echo "Error running the model! Exiting..."
        exit(1)
    endif

    # Then get ground truth
    set VAL_ARGS = "$CONFIG_FILEPATH  $CHKPT_IDX ${DATA_DIR} --split $SPLIT --n_passes $n_passes --vertical_level $VERTICAL_LEVEL"

    echo "python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS"

#    python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS

    # Post process output
    set PP_ARGS = "--exp_dir $EXP_DIR --exp_name $EXP_NAME --config_filepath $CONFIG_FILEPATH --meta_filepath $META_FILEPATH --n_passes $n_passes --vertical_level $VERTICAL_LEVEL --split $SPLIT --mode validate"

    echo "python3 -m NASA_AQcGAN.scripts.postprocess_predictions $PP_ARGS"
    python3 -m NASA_AQcGAN.scripts.postprocess_predictions $PP_ARGS

    # Next pass
    @ n_passes++
end
