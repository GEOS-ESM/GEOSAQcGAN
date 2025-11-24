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

set GEOS_AQCGAN_CONFIG = $1
source $GEOS_AQCGAN_CONFIG

set argv = ()

# set environment
setenv PYTHONPATH ${SRC_DIR}/install/lib/Python
source $SRC_DIR/env@/g5_modules


# Make directories to hold the input and output data
mkdir -p ${DATA_DIR}
mkdir -p ${EXP_DIR}

# copy model norm stats file
# it will be edited by the preprocess script
cp $NORM_STATS_FILENAME ${DATA_DIR}

if ( $PREPROCESS_DATA == 1) then
    mkdir -p ${DATA_DIR}/train
    mkdir -p ${DATA_DIR}/test
    mkdir -p ${DATA_DIR}/val
    
    if ( $GEOS_DATA == "TrainingEnsemble" ) then
        # make soft links to GEOS ensemble data on /css
        ./link_train_ens_data.j $GEOS_AQCGAN_CONFIG $PREPROC_CONFIG_FILEPATH

        # preprocess GEOS data to make input files for AQcGAN
        ./preprocess_geos_ens_data.j $GEOS_AQCGAN_CONFIG $PREPROC_CONFIG_FILEPATH

    else
        set beg_date = `awk '/beg_date:/ {print $2}' $PREPROC_CONFIG_FILEPATH`
        set end_date = `awk '/end_date:/ {print $2}' $PREPROC_CONFIG_FILEPATH`

        rm -f ${DATA_DIR}/${SPLIT}/*${beg_date}-${end_date}.fields.npy
        rm -f ${DATA_DIR}/${SPLIT}/*${beg_date}-${end_date}.time.npy
        rm -f ${DATA_DIR}/*${beg_date}-${end_date}.meta.pkl

        # run preprocess script
        set PRE_ARGS = "--data_dir $DATA_DIR --geos_cf_yaml_file $PREPROC_CONFIG_FILEPATH --aqcgan_config_file $AQCGAN_CONFIG_FILEPATH --split $SPLIT --level $VERTICAL_LEVEL"

        echo "python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf $PRE_ARGS"
        python3 -m NASA_AQcGAN.scripts.preprocess_geos_cf $PRE_ARGS

        # Symbolic links to names the code expects.  Legacy - needs to be refactored.
        cp ${DATA_DIR}/${SPLIT}/*${beg_date}-${end_date}.fields.npy ${DATA_DIR}/${SPLIT}/1.npy
        cp ${DATA_DIR}/${SPLIT}/*${beg_date}-${end_date}.time.npy ${DATA_DIR}/${SPLIT}/1_time.npy
        cp ${DATA_DIR}/*${beg_date}-${end_date}.meta.pkl ${DATA_DIR}/meta.pkl

    endif

    ln -sf ${DATA_DIR}/${SPLIT}/1.npy ${DATA_DIR}/train/1.npy
    ln -sf ${DATA_DIR}/${SPLIT}/1_time.npy ${DATA_DIR}/train/1_time.npy

    if ( $SPLIT == "test") then
        set split_other = "val"
    else
        set split_other = "test"
    endif
    
    ln -sf ${DATA_DIR}/${SPLIT}/1.npy ${DATA_DIR}/${split_other}/1.npy
    ln -sf ${DATA_DIR}/${SPLIT}/1_time.npy ${DATA_DIR}/${split_other}/1_time.npy

    # link over AQcGAN model files (pickle and checkpoint files) to exp_dir
    if ( ! -f ${EXP_DIR}/train_metrics.pkl ) then
        ln -s ${MODEL_ROOT}/train_metrics.pkl ${EXP_DIR}
    endif
    if ( ! -f ${EXP_DIR}/val_metrics.pkl ) then
        ln -s ${MODEL_ROOT}/val_metrics.pkl ${EXP_DIR}
    endif

    if ( ! -d ${EXP_DIR}/${MODEL_DIR} ) then
        ln -s ${MODEL_DIR} ${EXP_DIR}
    endif

endif

#######################################################################
#          STEP 2: Run the NASA-AQcGAN Model in inference mode 
#######################################################################
if ( $CLEAN_PREV_OUTPUT == 1 ) then
    rm -f ${EXP_DIR}/aqcgan_predictions/*.nc4
    rm -f ${EXP_DIR}/*stats*.npz
endif

if ( $RUN_AQCGAN == 1 ) then
    set n_passes=1
    while ( $n_passes <= $MAX_N_PASSES )
        # Get predictions first
        set VAL_ARGS = "$AQCGAN_CONFIG_FILEPATH  $CHKPT_IDX ${DATA_DIR} --split $SPLIT --n_passes $n_passes --vertical_level $VERTICAL_LEVEL --is_pred"

        echo "python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS"
        python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS

        if ($? != 0) then
            echo "Error running the model! Exiting..."
            exit(1)
        endif

        if ( $MODE == "validate" ) then
            # Then get ground truth
            set VAL_ARGS = "$AQCGAN_CONFIG_FILEPATH  $CHKPT_IDX ${DATA_DIR} --split $SPLIT --n_passes $n_passes --vertical_level $VERTICAL_LEVEL"

            echo "python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS"
            python3 -m NASA_AQcGAN.inference.create_ensemble_predictions $VAL_ARGS
        endif

        # Post process output
        set PP_ARGS = "--exp_dir $EXP_DIR --exp_name $EXP_NAME --config_filepath $AQCGAN_CONFIG_FILEPATH --meta_filepath ${DATA_DIR}/meta.pkl --n_passes $n_passes --vertical_level $VERTICAL_LEVEL --split $SPLIT --mode $MODE"

        echo "python3 -m NASA_AQcGAN.scripts.postprocess_predictions $PP_ARGS"
        python3 -m NASA_AQcGAN.scripts.postprocess_predictions $PP_ARGS

        # Next pass
        @ n_passes++
    end
endif
