import argparse
from pathlib import Path

import numpy as np

from lib import FinnDir, finn_fit_retardation, load_exp_conf, load_exp_data

data_core2_df = load_exp_data(name="Core 2")
conf_core2 = load_exp_conf(name="Core 2")


def run_experiment(finn_dir: FinnDir, data):
    np.save(finn_dir.c_train_path, np.squeeze(data))
    t = np.linspace(data_core2_df["time"].min(), data_core2_df["time"].max(), len(data))
    np.save(finn_dir.t_train_path, t)

    print(f"{finn_dir.path} is ready to run")

    rng = np.random.default_rng()

    finn_fit_retardation(
        out_dir=finn_dir.path,
        is_exp_data=True,
        n_epochs=int(rng.integers(low=10, high=31)),
        random_hypers=True,
        **conf_core2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FINN with restart logic.")
    parser.add_argument("finn_dir", type=Path)
    parser.add_argument("data_file", type=Path)
    args = parser.parse_args()

    # Run the experiment with the provided parameters
    run_experiment(FinnDir(args.finn_dir), np.load(args.data_file))
