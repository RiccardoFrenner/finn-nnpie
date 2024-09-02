"""
Creates the PI datasets for the FINN models (+, -) by using the mean predictions, median shifts and residual net predictions.
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_dataset(mode, quantile, ret_type):
    base_dir = Path(f"data_out/{ret_type}/default_finn").resolve()
    Nt = json.loads((base_dir / "input.json").read_text())["train_split_idx"]
    Nx = 26  # TODO
    res_net_out_path = base_dir / "residual_nets_output"

    full_residuals_diss = np.load(res_net_out_path / f"predictions_{mode}_diss.npy")
    full_residuals_tot = np.load(res_net_out_path / f"predictions_{mode}_tot.npy")
    # print(
    #     f"Min/Max {mode} residual diss: {full_residuals_diss.min()} {full_residuals_diss.max()}"
    # )
    # print(
    #     f"Min/Max {mode} residual tot: {full_residuals_tot.min()} {full_residuals_tot.max()}"
    # )

    Y_finn_mean_diss = np.load(base_dir / "c_train_predictions.npy")[:, 0, ...].reshape(
        (-1, 1)
    )
    Y_finn_mean_tot = np.load(base_dir / "c_train_predictions.npy")[:, 1, ...].reshape(
        (-1, 1)
    )
    residual_medians = np.load(base_dir / "residual_medians.npy")
    Y_finn_diss = full_residuals_diss + Y_finn_mean_diss + residual_medians[0]
    Y_finn_tot = full_residuals_tot + Y_finn_mean_tot + residual_medians[1]
    # Y_finn_diss = full_residuals_diss
    # Y_finn_tot = full_residuals_tot

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    fig.suptitle(f"{mode} FINN PI datasets")
    pcolor_shape = (Nt, Nx)
    ax1.pcolor(Y_finn_diss.reshape(pcolor_shape).T)
    ax2.pcolor(Y_finn_tot.reshape(pcolor_shape).T)

    Y_finn_diss = Y_finn_diss.reshape(pcolor_shape)[:, np.newaxis, ...].copy()
    Y_finn_tot = Y_finn_tot.reshape(pcolor_shape)[:, np.newaxis, ...].copy()

    # load training data for mean network
    Y_train = np.load(base_dir / "c_train.npy")
    Y_train_diss = Y_train[:, 0, ...][:, np.newaxis, ...]
    Y_train_tot = Y_train[:, 1, ...][:, np.newaxis, ...]

    # shift PI datasets
    if mode == "pos":
        Y_finn_diss += np.quantile(Y_train_diss - Y_finn_diss, quantile)
        Y_finn_tot += np.quantile(Y_train_tot - Y_finn_tot, quantile)
        # count how much percent of Y_train is below Y_finn
        below_diss = (Y_train_diss < Y_finn_diss).sum() / Y_train_diss.size
        below_tot = (Y_train_tot < Y_finn_tot).sum() / Y_train_tot.size
        print(f"{mode} below diss: {below_diss*100:.1f}")
        print(f"{mode} below tot: {below_tot*100:.1f}")
        assert below_diss >= quantile * 0.9 and below_tot >= quantile * 0.9, (
            below_diss,
            below_tot,
            quantile,
        )
    elif mode == "neg":
        Y_finn_diss -= np.quantile(Y_finn_diss - Y_train_diss, quantile)
        Y_finn_tot -= np.quantile(Y_finn_tot - Y_train_tot, quantile)
        # count how much percent of Y_train is above Y_finn
        above_diss = (Y_train_diss > Y_finn_diss).sum() / Y_train_diss.size
        above_tot = (Y_train_tot > Y_finn_tot).sum() / Y_train_tot.size
        print(f"{mode} above diss: {above_diss*100:.1f}")
        print(f"{mode} above tot: {above_tot*100:.1f}")
        assert above_diss >= quantile * 0.9 and above_tot >= quantile * 0.9, (
            above_diss,
            above_tot,
            quantile,
        )
    else:
        raise ValueError("mode must be 'pos' or 'neg'")

    Y_finn = np.concatenate([Y_finn_diss, Y_finn_tot], axis=1)

    out_dir = base_dir / "finn_stds_input"
    out_dir.mkdir(exist_ok=True, parents=True)
    np.save(out_dir / f"Y_{mode}_finn_quantile={int(quantile*100)}.npy", Y_finn)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("quantile", type=float)
    parser.add_argument("--ret_type", type=str, default="langmuir")
    args = vars(parser.parse_args())

    assert args["quantile"] > 0.5 and args["quantile"] < 1.0, args["quantile"]

    make_dataset(mode="pos", **args)
    make_dataset(mode="neg", **args)
