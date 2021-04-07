#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -n 1

set -x

module load 2020
. ~/qaoa/bin/activate

"$HOME/bachelor-thesis/scripts/qaoa_maxcut.py" \
       	--qi-backend-type='Starmon-5'          \
	--max-iter=30 \
	--optimizer='spsa' \
	--shots=4096 \
	--solution-shots=8192 \
	--layers=2 \
	--seed=334 \
       	--output-file="$HOME/qaoa_maxcut_n4_p2_starmon_spsa.npz"
