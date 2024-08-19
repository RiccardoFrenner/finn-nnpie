from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


class AnalyticRetardation:
    @staticmethod
    def linear(u, por, rho_s, Kd):
        factor = 1 + (1 - por) / por * rho_s * Kd
        ones_like_u = u * 0 + 1
        return ones_like_u * factor

    @staticmethod
    def freundlich(u, por, rho_s, Kf, nf):
        return 1 + (1 - por) / por * rho_s * Kf * nf * (u + 1e-6) ** (nf - 1)

    @staticmethod
    def langmuir(u, por, rho_s, smax, Kl):
        return 1 + (1 - por) / por * rho_s * smax * Kl / ((u + Kl) ** 2)


def create_mlp(layers: list[int], activation_fun, activation_fun_end):
    network_layers = []

    for i in range(len(layers) - 1):
        network_layers.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            network_layers.append(activation_fun)

    network_layers.append(activation_fun_end)

    return nn.Sequential(*network_layers)


def plot_c_spaceseries(c, t_idx):
    plt.figure(figsize=(10, 5))
    plt.plot(c[t_idx, 0, :], label="diss")
    plt.plot(c[t_idx, 1, :], label="tot")
    plt.title(f"Array at time step {t_idx}")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.legend()


def plot_c_timeseries(c, x_idx):
    plt.figure(figsize=(10, 5))
    plt.plot(c[:, 0, x_idx], label="diss")
    plt.plot(c[:, 1, x_idx], label="tot")
    plt.title(f"c at x_idx = {x_idx}")
    plt.xlabel("Time step")
    plt.ylabel("c")
    plt.legend()


def plot_c(t, x, c: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
    pcolor_plot = ax1.pcolor(t, x, c[:, 0, :], cmap="viridis")
    plt.colorbar(pcolor_plot)
    ax1.set_title("c_diss")
    ax1.set_xlabel("X index")
    ax1.set_ylabel("Time step")

    pcolor_plot = ax2.pcolor(t, x, c[:, 1, :], cmap="viridis")
    plt.colorbar(pcolor_plot)
    ax2.set_title("c_tot")
    ax2.set_xlabel("X index")
    ax2.set_ylabel("Time step")


def pcolor_from_scatter(x, y, z, placeholder_value=0):
    # Generate grid points from unique x and y values
    unique_x = sorted(set(x))
    unique_y = sorted(set(y))
    X, Y = np.meshgrid(unique_x, unique_y)

    # Create a dictionary to store z values based on (x, y) coordinates
    z_dict = {(x_val, y_val): z_val for x_val, y_val, z_val in zip(x, y, z)}

    # Create an array to store the z values for the pcolor plot
    Z = np.zeros_like(X)

    # Fill the Z array with actual and placeholder z values
    for i, x_val in enumerate(unique_x):
        for j, y_val in enumerate(unique_y):
            Z[j, i] = z_dict.get((x_val, y_val), placeholder_value)

    # Create the pcolor plot
    plt.pcolor(X, Y, Z)
    plt.colorbar()


def iter_final_retardation_files(
    root,
    min_epoch: int=100,
    is_ret_OK: Optional[Callable[[np.ndarray], bool]] = None,
    verbose: bool = False,
):
    """
    Iterate trough a directory containing multiple
    folders with FINN simulation results.
    Return the path to the final ret curve file.
    """
    finn_dirs = (p.parent for p in root.rglob("c_predictions.npy"))
    for p in finn_dirs:
        all_ret_file_paths = sorted(
            (p / "predicted_retardations").glob("retPred_*.npy"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if not all_ret_file_paths:
            if verbose:
                print(f"Skipped {p} because no files found")
            continue

        epoch = int(all_ret_file_paths[-1].stem.split("_")[-1])
        if epoch < min_epoch:
            if verbose:
                print(f"Skipped {p} because epoch < {min_epoch}")
            continue

        ret = np.load(all_ret_file_paths[-1])
        if np.any(np.isnan(ret)):
            if verbose:
                print(f"Skipped {p} because ret contains NaNs")
            continue

        if np.any(np.isinf(ret)):
            if verbose:
                print(f"Skipped {p} because ret contains infs")
            continue

        if np.any(ret > 1e6) or np.any(ret < 1e-6):
            if verbose:
                print(f"Skipped {p} because ret is not in [1e-6, 1e6]")
            continue

        if is_ret_OK is not None:
            if not is_ret_OK(ret):
                if verbose:
                    print(f"Skipped {p} because ret is not OK")
                continue

        yield p, all_ret_file_paths[-1]


def is_below_curve(
    curve_x: np.ndarray, curve_y: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Checks whether a set of points is below a curve. The curve is a piecewise linear function given by points.

    Args:
        curve_x (np.ndarray): x-coordinates of the curve points
        curve_y (np.ndarray): y-coordinates of the curve points
        points (np.ndarray): (N, 2) array containing the points to check

    Returns:
        np.ndarray: Boolean array indicating whether each point is below the curve.
    """
    # Interpolate the curve using piecewise linear interpolation
    interpolated_y = np.interp(points[:, 0], curve_x, curve_y)

    # Compare the y-coordinates of the points with the interpolated y-values of the curve
    below_curve = points[:, 1] < interpolated_y

    return below_curve


def is_above_curve(
    curve_x: np.ndarray, curve_y: np.ndarray, points: np.ndarray
) -> np.ndarray:
    interpolated_y = np.interp(points[:, 0], curve_x, curve_y)
    below_curve = points[:, 1] > interpolated_y
    return below_curve
