#!/bin/bash


time_steps=6
y_train_path="data/synthetic_data/retardation_freundlich/c_train.npy"

for ((i=0; i<3; i++))
do
  output_dir="data_out/finn_increasing_time_${time_steps}"
  mkdir -p $output_dir
  echo "Running Finn on time steps ${time_steps} inside ${output_dir}"
  echo "------------------------------------"
  echo "------------------------------------"

  python src/train_finn.py $y_train_path $output_dir -s $time_steps

  time_steps=$((time_steps + 30))
done
