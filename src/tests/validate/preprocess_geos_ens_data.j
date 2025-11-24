#!/bin/csh -fx

#######################################################################
#                     Batch Parameters for Run Job
#######################################################################
#SBATCH -J aqcgan_merge
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -A s2825
#SBATCH -o output_aqcgan_merge-%j.log

#######################################################################
#                   Preprocess GEOS ensemble data
#######################################################################
set aqcgan_config = $1
source $aqcgan_config

set preproc_config = $2
source $preproc_config

set argv = ()

set SRC_DIR = /gpfsm/dnb06/projects/p271/vshah5/GEOSAQcGAN
setenv PYTHONPATH ${SRC_DIR}/install/lib/Python
source $SRC_DIR/env@/g5_modules

#######################################################################
#                           Settings
#######################################################################
set members1 = ()
foreach element ($members)
    set members1 = ($members1 `expr $element + 0`)
end

if ($SPLIT == "val") then
    set val_members = ($members1)
    set test_members = 99
else
    set test_members = ($members1)
    set val_members = 99
endif

set train_members = 99

set data_type=gcc

#######################################################################
#                 Process members
#######################################################################
foreach mem ($members1)
    echo "Processing member $mem..."
    set args="$perturb $data_type $mem --data_dir ${DATA_DIR}/GEOS_output --save_dir $DATA_DIR --train_members $train_members --test_members $test_members --val_members $val_members"
    python3 -m NASA_AQcGAN.scripts.merge_geos_cf_ens_data $args &
end

wait
