#!/bin/bash


python_script="src/train_finn.py"


# step=51
# iterations=6
# for i in $(seq 1 $iterations); do
#   output_dir="data_out/finn_running_interval_${i}"
#   y_train_path="data/synthetic_data/retardation_freundlich/sub_intervals/c_${i}.npy"

#   start=$(($step * $i))
#   end=$(($start + $step))

#   echo "Running Python script for time steps $start:$end..."

#   python "$python_script" "$y_train_path" "$output_dir" --train_split_idx "$end" --skip "$start" 

#   start=$((end))
# done




# gnu parallel version


python_script="src/train_finn.py"
step=51

export python_script y_train_path step

seq 0 65 | parallel '
  start=$(($step * {}))
  end=$(($start + $step))
  output_dir="data_out/finn_running_intervas_stepsize_30/finn_running_interval_{}"
  y_train_path="data/synthetic_data/retardation_freundlich/sub_intervals/c_{}.npy"
  echo "Running Python script for time steps $start:$end..."
  python "$python_script" "$y_train_path" "$output_dir" --train_split_idx 30 --skip 0
'