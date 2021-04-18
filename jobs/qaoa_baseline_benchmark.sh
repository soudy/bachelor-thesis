#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -n 1

set -x

module load 2020
. ~/qaoa/bin/activate

max_iter=30
optimizer="COBYLA"
shots=4096
layers=2
seed=11191001

for i in {1..5}; do
  out_file="$HOME/qaoa_baseline/${1}/${i}.npz"
  echo "Out file: $out_file"

  if [[ "$1" = "local" ]]; then
    "$HOME/bachelor-thesis/scripts/qaoa_maxcut.py" \
      --max-iter=$max_iter \
      --optimizer=$optimizer \
      --shots=$shots \
      --layers=$layers \
      --seed=$seed \
      --output-file=$out_file
  elif [[ "$1" = "qpu" ]]; then
    "$HOME/bachelor-thesis/scripts/qaoa_maxcut.py" \
      --qi-backend-type='Starmon-5'          \
      --max-iter=$max_iter \
      --optimizer=$optimizer \
      --shots=$shots \
      --layers=$layers \
      --seed=$seed \
      --output-file=$out_file
  elif [[ "$1" = "hpc" ]]; then
    "$HOME/bachelor-thesis/scripts/qaoa_maxcut.py" \
      --qi-backend-type='QX-31-L'          \
      --qi-api-url='https://staging.quantum-inspire.com' \
      --max-iter=$max_iter \
      --optimizer=$optimizer \
      --shots=$shots \
      --layers=$layers \
      --seed=$seed \
      --output-file=$out_file
  else
    echo "Invalid argument $1"
    exit 1
  fi
done
