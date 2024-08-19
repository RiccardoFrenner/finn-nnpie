default_c_train="data/synthetic_data/retardation_freundlich/c_train.npy"

# TODO: Why 51 and not 251 as in paper?


# train finn with default parameters. used for residual training data to perform 3pinn on that and just to have the data for plots to compare against
# echo "Train finn with default parameters"
# python src/train_finn.py ${default_c_train} "data_out/default_finn" -s 51 --skip 0 --max_epochs 100 --seed 564345


# echo "Generate 3pinn residual training data for C"
# ./run_notebook.sh src/make_c_residual_data.ipynb


# FIXME: Does somehow not work. Running these manually leads to conversion. But not if run like this.
# echo "Train residual networks"
# export C_TYPE="tot"
# export C_MODE="pos"
# ./run_notebook.sh src/train_residual_net_tf.ipynb
# export C_TYPE="diss"
# export C_MODE="pos"
# ./run_notebook.sh src/train_residual_net_tf.ipynb
# export C_TYPE="tot"
# export C_MODE="neg"
# ./run_notebook.sh src/train_residual_net_tf.ipynb
# export C_TYPE="diss"
# export C_MODE="neg"
# ./run_notebook.sh src/train_residual_net_tf.ipynb


# echo "Generate 3pinn std network training data for different quantiles"
# for quantile in 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
# do
#     python src/make_c_PI_datasets.py ${quantile}
# done


# train negative and positive c quantiles
# echo "Train std finns"
# python src/train_std_finns.py

# echo "Train finns with different seeds"
# python src/train_finn_different_seeds.py

echo "Train finn running intervals"
python src/train_finn_running_intervals.py --max_epochs 1000

# echo "Train finn increasing time"
# python src/train_finn_increasing_time.py

# echo "Train finns with different loss patterns"
# python src/train_finn_different_loss_patterns.py
