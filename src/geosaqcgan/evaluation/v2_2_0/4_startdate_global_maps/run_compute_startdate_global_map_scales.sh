#!/bin/bash

#SBATCH --job-name=global_map_scales
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=global_map_scales_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Script Directory  = $SCRIPT_DIR"
echo "SLURM Job ID      = ${SLURM_JOB_ID:-N/A}"
echo ""

module load python/GEOSpyD/24.11.3-0/3.12

cd "$SCRIPT_DIR"

python -u "$SCRIPT_DIR/compute_startdate_global_map_scales.py"
