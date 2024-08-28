analytical_ret="langmuir"

base_in_dir="data/FINN_forward_solver/retardation_${analytical_ret}"
default_c_train="${base_in_dir}/c_train.npy"
base_out_dir="data_out"

max_epochs=10

# TODO: Why 51 and not 251 as in paper?


# train finn with default parameters. used for residual training data to perform 3pinn on that and just to have the data for plots to compare against
# echo "Train finn with default parameters"
# python src/train_finn.py ${default_c_train} "data_out/${analytical_ret}/default_finn" -s 51 --skip 0 --max_epochs ${max_epochs} --seed 564345


# echo "Generate 3pinn residual training data for C"
# python src/make_c_residual_data.py --ret_type ${analytical_ret}


# echo "Train residual networks"
# parallel --bar python src/train_residual_net_tf.py --ret_type ${analytical_ret} --mode {1} --c_type {2} ::: pos neg ::: tot diss


# echo "Generate 3pinn std network training data for different quantiles"
# parallel --bar python src/make_c_PI_datasets.py {} --ret_type ${analytical_ret} ::: 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95


# train negative and positive c quantiles
echo "Train std finns"
python src/train_std_finns.py --ret_type ${analytical_ret} --max_epochs ${max_epochs}

# echo "Train finns with different seeds"
# python src/train_finn_different_seeds.py --ret_type ${analytical_ret} --max_epochs ${max_epochs}

# echo "Train finn running intervals"
# python src/train_finn_running_intervals.py --ret_type ${analytical_ret} --max_epochs ${max_epochs}

# echo "Train finn increasing time"
# python src/train_finn_increasing_time.py --ret_type ${analytical_ret} --max_epochs ${max_epochs}

echo "Train finns with different loss patterns"
python src/train_finn_different_loss_patterns.py --ret_type ${analytical_ret} --max_epochs ${max_epochs}

# echo "Train finn with first, second and third running interval but for many seeds to see if the latter converge at all"
# for i in 0 1 2; do
#     echo $i$/2
#     seeds=$(for j in $(seq 1 5); do echo $RANDOM; done | xargs)
#     parallel -j 8 --bar python src/train_finn.py ${base_in_dir}/sub_intervals/c_${i}.npy ${base_out_dir}/finn_first_running_intervals_stepsize_30_epochs_1000/c_${i}_seed_{} --train_split_idx 30 --seed {} --max_epochs 1000 ::: ${seeds}
# done

# echo "Train FINN with c plus noise"
# python src/train_finn_c_plus_noise.py --ret_type ${analytical_ret} --max_epochs ${max_epochs}

# echo "Train FINN with dropout"
# python src/train_finn.py ${default_c_train} "data_out/finn_with_dropout/p=10" -s 51 --skip 0 --max_epochs ${max_epochs} --seed 2134834 --dropout 10
# python src/train_finn.py ${default_c_train} "data_out/finn_with_dropout/p=50" -s 51 --skip 0 --max_epochs ${max_epochs} --seed 3485637 --dropout 50
# python src/train_finn.py ${default_c_train} "data_out/finn_with_dropout/p=90" -s 51 --skip 0 --max_epochs ${max_epochs} --seed 9837432 --dropout 90

# echo "Train FINN with github data and self-generated data to compare both"
# parallel --bar python src/train_finn.py "data/synthetic_data/FINN_forward_solver/retardation_{}/c_train.npy" "data_out/FINN_forward_tests/finn_{}_selfgen_c" -s 51 --skip 0 --max_epochs ${max_epochs} --seed 123456 ::: langmuir freundlich linear
# parallel --bar python src/train_finn.py "data/synthetic_data/retardation_{}/c_train.npy" "data_out/FINN_forward_tests/finn_{}_github_c" -s 51 --skip 0 --max_epochs ${max_epochs} --seed 123456 ::: langmuir freundlich linear

# echo "Run a few langmuir finns with github data to check if they all look so different from the langmuir ret although having a small c error"
# parallel --bar -j 8 python src/train_finn.py "data/synthetic_data/retardation_langmuir/c_train.npy" "data_out/FINN_forward_tests/finn_langmuir_github_c_{}" -s 51 --skip 0 --max_epochs ${max_epochs} --seed {} ::: 3546 4385763 238479 98789354 626734 37264333 598567 554242

# echo "Run a few freundlich finns with selfgen data to check if they all stop before max_epochs despite looking good (in retardation)"
# parallel --bar -j 8 python src/train_finn.py "data/synthetic_data/FINN_forward_solver/retardation_freundlich/c_train.npy" "data_out/FINN_forward_tests/finn_freundlich_selfgen_c_{}" -s 51 --skip 0 --max_epochs ${max_epochs} --seed {} ::: 3546 4385763 238479 98789354 626734 37264333 598567 554242

say "Done"