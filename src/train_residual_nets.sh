#!/bin/bash

input_dir="data_out/residual_training_data"
output_dir="data_out/residual_nets_output"

for target in pos neg; do
  for feature in tot diss; do
  echo "Training residual net for ${target} ${feature}"
    python src/train_residual_net_tf.py \
      "${input_dir}/X_${target}_train_${feature}.npy" \
      "${input_dir}/Y_${target}_train_${feature}.npy" \
      "${input_dir}/X_${target}_test_${feature}.npy" \
      "${output_dir}/Y_${target}_test_${feature}.npy"
  done
done