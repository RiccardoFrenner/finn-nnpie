import argparse
import os
from pathlib import Path

import numpy as np

from lib import FinnDir, finn_fit_retardation, load_exp_conf, load_exp_data

data_core2_df = load_exp_data(name="Core 2")
conf_core2 = load_exp_conf(name="Core 2")


def check_convergence_and_restart(finn_dir: FinnDir):
    # Function to check convergence and handle restarts
    mse_threshold = 4e-6

    # Check if experiment is done
    if finn_dir.is_done:
        # Load the predicted and training data
        pred_data = np.load(finn_dir.get_data_pred_path(finn_dir.best_epoch))
        train_data = np.load(finn_dir.c_train_path)

        # Calculate MSE
        mse = np.mean((pred_data - train_data) ** 2)
        rel_mse = mse / train_data.max()

        # Check if MSE is below the threshold
        if rel_mse < mse_threshold:
            print(f"Convergence achieved for {finn_dir.path}. rel. MSE: {rel_mse}")
            return True  # Convergence achieved
        else:
            # If done but not converged, delete done marker and retry
            print(
                f"Experiment finished but did not converge for {finn_dir.path}. Restarting..."
            )
            os.remove(finn_dir.done_marker_path)

    # Run the experiment
    print(f"Running experiment for {finn_dir.path}")
    finn_fit_retardation(
        out_dir=finn_dir.path,
        is_exp_data=True,
        n_epochs=21,
        **conf_core2,
    )
    return False  # Not converged


def run_experiment(finn_dir: FinnDir, data):
    np.save(finn_dir.c_train_path, np.squeeze(data))
    t = np.linspace(data_core2_df["time"].min(), data_core2_df["time"].max(), len(data))
    np.save(finn_dir.t_train_path, t)

    print(f"{finn_dir.path} is ready to run")

    # Restart loop until convergence is achieved
    max_restarts = 4
    converged = False
    for _ in range(max_restarts):
        if converged:
            break
        converged = check_convergence_and_restart(finn_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FINN with restart logic.")
    parser.add_argument("finn_dir", type=Path)
    parser.add_argument("data_file", type=Path)
    args = parser.parse_args()

    # Run the experiment with the provided parameters
    run_experiment(FinnDir(args.finn_dir), np.load(args.data_file))
