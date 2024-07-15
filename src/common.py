import numpy as np
import torch
import torch.nn as nn


class EarlyStopper:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def update(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopper counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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


import matplotlib.pyplot as plt


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
