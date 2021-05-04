#! /bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -c 40
#SBATCH --mem 1400G
#SBATCH -p fat_soil_shared

set -x

module load 2020
module use ~/environment/modules/QX/
module load ~/environment/modules/QX/b351996-v0.3.0-intel

SCRIPT_PATH="$HOME/bachelor-thesis/scripts/qx_benchmark.py"
DATA_DIR="$HOME/bachelor-thesis/data/qx_benchmark_cqasm"
OUT_DIR="$HOME/qx_benchmark_fat_soil_shared"

cores=(40 36 32 24 20 16 8 4 2 1)
ns=(4 6 8 10 12 14 16 18 20 22 24 26 28 30)

for core in "${cores[@]}"; do
    for n in "${ns[@]}"; do
        python $SCRIPT_PATH \
	       	"$OUT_DIR/qaoa_n${n}_cores${core}.npz" \
          "$DATA_DIR/qaoa_n${n}_p1_d2.qc" \
          $core \
          --reps=10
    done
  done
