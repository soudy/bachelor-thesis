#! /bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -c 40
#SBATCH --mem 1400G
#SBATCH -p fat_soil_shared

set -x

module load 2020
module use ~/environment/modules/QX/
module load ~/environment/modules/QX/b351996-v0.3.0-intel

python $HOME/bachelor-thesis/scripts/qx_memory_usage.py
