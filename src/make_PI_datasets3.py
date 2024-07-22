"""
Creates the PI datasets for the FINN models (+, -) by using the mean predictions plus noise.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_dataset(mode="pos"):
    base_dir = Path("data_out").resolve()

    Y_finn_mean_diss = np.load(base_dir / "c_predictions.npy")[:, 0, ...]
    Y_finn_mean_tot = np.load(base_dir / "c_predictions.npy")[:, 1, ...]

    NOISE_SIGMA = 1e-4
    noise_diss = np.random.normal(0, NOISE_SIGMA, Y_finn_mean_diss.shape)
    noise_tot = np.random.normal(0, NOISE_SIGMA, Y_finn_mean_diss.shape)
    if mode == "pos":
        noise_diss *= np.sign(noise_diss)
        noise_tot *= np.sign(noise_tot)
        assert np.all(noise_diss > 0) and np.all(noise_tot > 0)
    else:
        noise_diss *= -np.sign(noise_diss)
        noise_tot *= -np.sign(noise_tot)
        assert np.all(noise_diss < 0) and np.all(noise_tot < 0)

    Y_finn_diss = Y_finn_mean_diss + noise_diss
    Y_finn_tot = Y_finn_mean_tot + noise_tot

    Y_finn_diss = Y_finn_diss[:, np.newaxis, ...]
    Y_finn_tot = Y_finn_tot[:, np.newaxis, ...]

    Y_finn = np.concatenate([Y_finn_diss, Y_finn_tot], axis=1)[..., 0]

    out_dir = base_dir / "finn_stds_input_noise"
    out_dir.mkdir(exist_ok=True, parents=True)
    np.save(out_dir / f"Y_{mode}_finn.npy", Y_finn)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    print(Y_finn.shape)
    fig.suptitle(f"{mode} Y_FINN")
    ax1.pcolor(Y_finn[:, 0, :].T)
    ax2.pcolor(Y_finn[:, 1, :].T)

    plt.show()


make_dataset("pos")
make_dataset("neg")
