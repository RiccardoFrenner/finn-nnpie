import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint

from lib_p3inn import (
    CL_trainer,
    UQ_Net_mean,
    UQ_Net_std,
    compute_boundary_factors,
    compute_shift_to_median,
    create_PI_training_data,
    shift_to_median,
)


class FinnParams:
    def __init__(
        self,
        *,
        # Training params
        n_epochs: int = 30,
        error_mult: float = 1e5,
        phys_mult: float = 1e2,
        start_lr: float = 1e-1,
        use_adam_optim: bool = False,
        # Domain params
        X: float = 1.0,  # length of sample [m]
        T: float = 10000,  # simulation time [days]
        Nx: int = 26,
        Nt: int = 201,
        # Soil params
        D: float = 0.0005,  # effective diffusion coefficient [m^2/day]
        por: float = 0.29,  # porosity [-]
        rho_s: float = 2880,  # bulk density [kg/m^3]
        solubility: float = 1.0,  # top boundary value [kg/m^3]
        # FINN params
        c_diss_max: float = 1.0,  # for evaluating retardation (inclusive; for phys loss and output)
        n_c_diss: int = 100,
        **kwargs,
    ):
        # Training params
        self.n_epochs = n_epochs
        self.error_mult = error_mult
        self.phys_mult = phys_mult
        self.start_lr = start_lr
        self.use_adam_optim = use_adam_optim

        # Domain params
        self.X = X
        self.T = T
        self.Nx = int(Nx)  # no. inner grid points (aka. no. cells - 1)
        self.Nt = int(Nt)
        self.dx = self.X / (self.Nx + 1)  # length of discrete control volume [m]
        self.dt = self.T / (
            self.Nt + 1
        )  # time step [days] # FIXME: Check if this is used
        ## Boundary conditions
        self.dirichlet_bool = kwargs.pop(
            "dirichlet_bool", [[True, False, False, False], [True, False, False, False]]
        )
        self.neumann_bool = kwargs.pop(
            "neumann_bool", [[False, False, True, True], [False, False, True, True]]
        )
        self.cauchy_bool = kwargs.pop(
            "cauchy_bool", [[False, True, False, False], [False, True, False, False]]
        )
        self.dirichlet_val = kwargs.pop(
            "dirichlet_val", [[solubility, 0.0, 0.0, 0.0], [solubility, 0.0, 0.0, 0.0]]
        )
        self.neumann_val = kwargs.pop(
            "neumann_val", [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        ## multiplier for the Cauchy boundary condition
        self.cauchy_mult = kwargs.pop("cauchy_mult", [self.dx, self.dx])

        # Soil params
        self.D = D
        self.por = por
        self.rho_s = rho_s
        self.solubility = solubility
        ## Effective diffusion coefficient for each variable
        self.D_eff = kwargs.pop("D_eff", [self.D / self.dx**2, 0.25])
        # self.D_eff = [self.D / self.dx**2, self.D * self.por / (self.rho_s/1000) / self.dx**2]
        # TODO: If I insert the params the above equation does not yield 0.25

        # FINN params
        self.c_diss_max = c_diss_max
        self.n_c_diss = n_c_diss
        ## Normalizer for functions that are approximated with a NN
        self.p_exp_flux = kwargs.pop("p_exp_flux", [0.0, 0.0])
        ## Whether diffusion coefficient is learnable or not
        self.learn_coeff = kwargs.pop("learn_coeff", [False, True])

        # check all kwargs where used
        assert len(kwargs) == 0, kwargs

    @classmethod
    def from_dict(cls, is_exp_data=False, **kwargs):
        if "Dirichlet" in kwargs:
            kwargs["dirichlet_bool"] = [
                [True, bool(kwargs.get("Dirichlet")), False, False],
                [True, bool(kwargs.pop("Dirichlet")), False, False],
            ]
        if "Cauchy" in kwargs:
            kwargs["cauchy_bool"] = [
                [False, bool(kwargs.get("Cauchy")), False, False],
                [False, bool(kwargs.pop("Cauchy")), False, False],
            ]

        if is_exp_data:
            kwargs["learn_coeff"] = [False, False]

        r = kwargs.pop("sample_radius")
        Q = kwargs.pop("Q")
        A = np.pi * r**2

        finn_params = cls(**kwargs)

        if is_exp_data:
            # TODO: Not sure why only for exp data
            finn_params.D_eff[1] = (
                finn_params.D
                * finn_params.por
                / (finn_params.rho_s / 1000)
                / (finn_params.dx**2)
            )

        cauchy_val = finn_params.por * A / Q * finn_params.dx
        finn_params.cauchy_mult = [cauchy_val, cauchy_val]

        return finn_params

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "error_mult": self.error_mult,
            "phys_mult": self.phys_mult,
            "start_lr": self.start_lr,
            "use_adam_optim": self.use_adam_optim,
            "X": self.X,
            "T": self.T,
            "Nx": self.Nx,
            "Nt": self.Nt,
            "dirichlet_bool": self.dirichlet_bool,
            "neumann_bool": self.neumann_bool,
            "cauchy_bool": self.cauchy_bool,
            "dirichlet_val": self.dirichlet_val,
            "neumann_val": self.neumann_val,
            "cauchy_mult": self.cauchy_mult,
            "D": self.D,
            "por": self.por,
            "rho_s": self.rho_s,
            "solubility": self.solubility,
            "D_eff": self.D_eff,
            "c_diss_max": self.c_diss_max,
            "n_c_diss": self.n_c_diss,
            "p_exp_flux": self.p_exp_flux,
            "learn_coeff": self.learn_coeff,
        }


def compute_flux(
    u_main,
    u_coupled,
    t: float,
    u0,
    Nx: int,
    Ny: int,
    D_eff: float,
    cauchy_mult: float,
    stencil: tuple[float, float],
    dirichlet_bool: list[bool],
    dirichlet_val: list[float],
    neumann_bool: list[bool],
    neumann_val: list[float],
    cauchy_bool: list[bool],
    cauchy_val: list[float],  # has to be list because this is modified in here
    coeff_nn=None,
    p_exp=None,
) -> torch.Tensor:
    """Computes the integrated flux between each control volume and its neighbors.

    Args:
        u_main (tensor): the unknown variable to be used to calculate the flux,
                         dim: [1, Nx, Ny]
        u_coupled (tensor): all necessary unknown variables required to calculate
                            the diffusion coeffient as a function,
                            dim: [num_features, Nx, Ny]
        t (float): time (scalar value, taken from the ODE solver)

    Returns:
        torch.Tensor: the integrated flux for all control volumes,
                      dim: [Nx, Ny]
    """
    # print(f"{t=}")
    # print(f"{u_main.flatten().tolist()=}")
    # print(f"{u_coupled.flatten().tolist()=}")

    # Reshape the input dimension for the coeff_nn model into [Nx, Ny, num_features]
    u_coupled = u_coupled.permute(1, 2, 0)

    # Calculate the flux multiplier (diffusion coefficient function) if set
    # to be a function, otherwise set as tensor of ones
    if coeff_nn is not None:
        assert p_exp is not None
        flux_mult = coeff_nn(u_coupled).squeeze(2) * 10**p_exp
    else:
        flux_mult = torch.ones(Nx, Ny)

    # print(flux_mult.flatten().tolist())

    # Squeeze the u_main dimension into [Nx, Ny]
    u_main = u_main.squeeze(0)

    # Left Boundary Condition
    if dirichlet_bool[0]:
        # If Dirichlet, calculate the flux at the boundary using the
        # Dirichlet value as a constant
        left_bound_flux = (
            (stencil[0] * dirichlet_val[0] + stencil[1] * u_main[0, :]).unsqueeze(0)
            * D_eff
            * flux_mult[0, :]
        )

    elif neumann_bool[0]:
        # If Neumann, set the Neumann value as the flux at the boundary
        left_bound_flux = torch.full((1, Ny), neumann_val[0])

    elif cauchy_bool[0]:
        # If Cauchy, first set the value to be equal to the initial condition
        # at t = 0.0, otherwise update the value according to the previous
        # time step value
        if t == 0.0:
            cauchy_val[0] = u0[0, :]
        else:
            cauchy_val[0] = (u_main[0, :] - cauchy_val[0]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        left_bound_flux = (
            (stencil[0] * cauchy_val[0] + stencil[1] * u_main[0, :]).unsqueeze(0)
            * D_eff
            * flux_mult[0, :]
        )

    # Calculate the fluxes of each control volume with its left neighboring cell
    left_neighbors = (
        (stencil[0] * u_main[:-1, :] + stencil[1] * u_main[1:, :])
        * D_eff
        * flux_mult[1:, :]
    )
    # Concatenate the left boundary fluxes with the left neighbors fluxes
    left_flux = torch.cat((left_bound_flux, left_neighbors))
    # print(f"{left_flux.flatten().tolist()=}")

    # Right Boundary Condition
    if dirichlet_bool[1]:
        # If Dirichlet, calculate the flux at the boundary using the
        # Dirichlet value as a constant
        right_bound_flux = (
            (stencil[0] * dirichlet_val[1] + stencil[1] * u_main[-1, :]).unsqueeze(0)
            * D_eff
            * flux_mult[-1, :]
        )

    elif neumann_bool[1]:
        # If Neumann, set the Neumann value as the flux at the boundary
        right_bound_flux = torch.full((1, Ny), neumann_val[1])

    elif cauchy_bool[1]:
        # If Cauchy, first set the value to be equal to the initial condition
        # at t = 0.0, otherwise update the value according to the previous
        # time step value
        if t == 0.0:
            cauchy_val[1] = u0[-1, :]
        else:
            cauchy_val[1] = (u_main[-1, :] - cauchy_val[1]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        right_bound_flux = (
            (stencil[0] * cauchy_val[1] + stencil[1] * u_main[-1, :]).unsqueeze(0)
            * D_eff
            * flux_mult[-1, :]
        )

    # Calculate the fluxes of each control volume with its right neighboring cell
    right_neighbors = (
        (stencil[0] * u_main[1:, :] + stencil[1] * u_main[:-1, :])
        * D_eff
        * flux_mult[:-1, :]
    )
    # Concatenate the right neighbors fluxes with the right boundary fluxes
    right_flux = torch.cat((right_neighbors, right_bound_flux))
    # print(f"{right_flux.flatten().tolist()=}")

    # Top Boundary Condition
    if dirichlet_bool[2]:
        # If Dirichlet, calculate the flux at the boundary using the
        # Dirichlet value as a constant
        top_bound_flux = (
            (stencil[0] * dirichlet_val[2] + stencil[1] * u_main[:, 0]).unsqueeze(1)
            * D_eff
            * flux_mult[:, 0]
        )

    elif neumann_bool[2]:
        # If Neumann, set the Neumann value as the flux at the boundary
        top_bound_flux = torch.full((Nx, 1), neumann_val[2])

    elif cauchy_bool[2]:
        # If Cauchy, first set the value to be equal to the initial condition
        # at t = 0.0, otherwise update the value according to the previous
        # time step value
        if t == 0.0:
            cauchy_val[2] = u0[:, 0]
        else:
            cauchy_val[2] = (u_main[:, 0] - cauchy_val[2]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        top_bound_flux = (
            (stencil[0] * cauchy_val[2] + stencil[1] * u_main[:, 0]).unsqueeze(1)
            * D_eff
            * flux_mult[:, 0]
        )

    # Calculate the fluxes of each control volume with its top neighboring cell
    top_neighbors = (
        (stencil[0] * u_main[:, :-1] + stencil[1] * u_main[:, 1:])
        * D_eff
        * flux_mult[:, 1:]
    )
    # Concatenate the top boundary fluxes with the top neighbors fluxes
    top_flux = torch.cat((top_bound_flux, top_neighbors), dim=1)
    # print(f"{top_flux.flatten().tolist()=}")

    # Bottom Boundary Condition
    if dirichlet_bool[3]:
        # If Dirichlet, calculate the flux at the boundary using the
        # Dirichlet value as a constant
        bottom_bound_flux = (
            (stencil[0] * dirichlet_val[3] + stencil[1] * u_main[:, -1]).unsqueeze(1)
            * D_eff
            * flux_mult[:, -1]
        )

    elif neumann_bool[3]:
        # If Neumann, set the Neumann value as the flux at the boundary
        bottom_bound_flux = torch.full((Nx, 1), neumann_val[3])

    elif cauchy_bool[3]:
        # If Cauchy, first set the value to be equal to the initial condition
        # at t = 0.0, otherwise update the value according to the previous
        # time step value
        if t == 0.0:
            cauchy_val[3] = u0[:, -1]
        else:
            cauchy_val[3] = (u_main[:, -1] - cauchy_val[3]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        bottom_bound_flux = (
            (stencil[0] * cauchy_val[3] + stencil[1] * u_main[:, -1]).unsqueeze(1)
            * D_eff
            * flux_mult[:, -1]
        )

    # Calculate the fluxes of each control volume with its bottom neighboring cell
    bottom_neighbors = (
        (stencil[0] * u_main[:, 1:] + stencil[1] * u_main[:, :-1])
        * D_eff
        * flux_mult[:, :-1]
    )
    # Concatenate the bottom neighbors fluxes with the bottom boundary fluxes
    bottom_flux = torch.cat((bottom_neighbors, bottom_bound_flux), dim=1)
    # print(f"{bottom_flux.flatten().tolist()=}")

    # Integrate all fluxes at all control volume boundaries
    flux = left_flux + right_flux + top_flux + bottom_flux

    # print(f"{flux.flatten().tolist()=}")
    # print()
    return flux


class Flux_Kernels(torch.nn.Module):
    def __init__(self, u0: torch.Tensor, cfg: FinnParams, var_idx: int, coeff_nn=None):
        super(Flux_Kernels, self).__init__()

        # Extracting the spatial dimension and initial condition of the problem
        # and store the initial condition value u0
        self.Nx = u0.size(0)
        self.Ny = u0.size(1)
        self.u0 = u0

        # Variables that act as switch to use different types of boundary
        # condition
        # Each variable consists of boolean values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to
        # be no-flux boundaries (zero neumann_val)
        self.dirichlet_bool = cfg.dirichlet_bool[var_idx]
        self.neumann_bool = cfg.neumann_bool[var_idx]
        self.cauchy_bool = cfg.cauchy_bool[var_idx]

        # Variables that store the values of the boundary condition of each type
        # Values = 0 if not used, otherwise specify in the configuration file
        # Each variable consists of real values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to
        # be no-flux boundaries
        self.dirichlet_val = cfg.dirichlet_val[var_idx]
        self.neumann_val = cfg.neumann_val[var_idx]

        # For Cauchy BC, the initial Cauchy value is set to be the initial
        # condition at each corresponding domain boundary, and will be updated
        # through time
        self.cauchy_val = [u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1]]

        # Set the Cauchy BC multiplier (to be multiplied with the gradient of
        # the unknown variable and the diffusion coefficient)
        self.cauchy_mult = cfg.cauchy_mult[var_idx]

        self.stencil = (1.0, -1.0)

        # Extract the diffusion coefficient scalar value and set to be learnable
        # if desired
        self.D_eff = cfg.D_eff[var_idx]
        if cfg.learn_coeff[var_idx]:
            self.D_eff = torch.nn.Parameter(
                torch.tensor([self.D_eff], dtype=torch.float)
            )  # type: ignore

        # Extract value of the normalizing constant to be applied to the output
        # of the NN that predicts the diffusion coefficient function
        self.p_exp = cfg.p_exp_flux[var_idx]

        # Initialize a NN to predict diffusion coefficient as a function of
        # the unknown variable if necessary
        if coeff_nn is not None:
            self.coeff_nn = coeff_nn
            self.p_exp = torch.nn.Parameter(
                torch.tensor([self.p_exp], dtype=torch.float)
            )  # type: ignore
        else:
            self.coeff_nn = None

    def forward(self, u_main, u_coupled, t):
        return compute_flux(
            u_main=u_main,
            u_coupled=u_coupled,
            t=t,
            u0=self.u0,
            Nx=self.Nx,
            Ny=self.Ny,
            D_eff=self.D_eff,  # type: ignore
            cauchy_mult=self.cauchy_mult,
            stencil=self.stencil,
            dirichlet_bool=self.dirichlet_bool,
            dirichlet_val=self.dirichlet_val,
            neumann_bool=self.neumann_bool,
            neumann_val=self.neumann_val,
            cauchy_bool=self.cauchy_bool,
            cauchy_val=self.cauchy_val,
            coeff_nn=self.coeff_nn,
            p_exp=self.p_exp,
        )


class Net_Model(torch.nn.Module):
    def __init__(self, u0: torch.Tensor, cfg: FinnParams, ret_funs: list):
        """Construct time derivative computer.

        Args:
            u0 (tensor): initial condition, dim: [num_features, Nx, Ny]
            cfg (dict): configuration object of the model setup,
                        containing boundary condition types, values,
                        learnable parameter settings, etc.
        """
        super(Net_Model, self).__init__()

        assert len(ret_funs) == 2

        self.cfg = cfg
        self.num_vars = u0.size(0)
        self.flux_modules = torch.nn.ModuleList(
            [
                Flux_Kernels(u0[i], self.cfg, i, coeff_nn=ret_funs[i])
                for i in range(self.num_vars)
            ]
        )

    def forward(self, t, u):
        """Compute du/dt.

        Args:
            t (float) : time, taken from the ODE solver
            u (tensor): the unknown variables to be calculated taken
                        from the previous time step,
                        dim: [num_features, Nx, Ny]

        Returns:
            torch.Tensor: the time derivative of u (du/dt),
                          dim: [num_features, Nx, Ny]
        """
        flux = [self.flux_modules[i](u[[0]], u[[0]], t) for i in range(self.num_vars)]
        dudt = torch.stack(flux)

        return dudt


def interp1D_torch(y, xmin, xmax, x):
    # TODO: Test this function
    y = y.reshape(-1)
    x = x.reshape(-1)

    n = len(y) - 1  # number of intervals
    xp = torch.linspace(xmin, xmax, n + 1)
    dx = (xmax - xmin) / n

    # Calculate the index of the interval
    i = torch.clip(((x - xmin) / dx).to(int), 0, n - 1)

    # Perform linear interpolation using broadcasting
    y_interp = y[i] + (y[i + 1] - y[i]) * (x - xp[i]) / dx

    return y_interp.reshape(-1, 1, 1)


def make_mlp():
    return nn.Sequential(
        nn.Linear(1, 15),
        nn.Tanh(),
        nn.Linear(15, 15),
        nn.Tanh(),
        nn.Linear(15, 15),
        nn.Tanh(),
        nn.Linear(15, 15),
        nn.Tanh(),
        nn.Linear(15, 1),
        nn.Sigmoid(),
    )


def solve_diffusion_sorption_pde(
    retardation_fun, t, finn_params: FinnParams, c0=None
) -> np.ndarray:
    if c0 is None:
        c0 = torch.zeros(2, finn_params.Nx, 1).to(torch.float32)

    def coeff_nn_fun(c):
        return 1 / retardation_fun(c)

    model = Net_Model(c0, finn_params, ret_funs=[coeff_nn_fun, None])
    model.eval()
    return odeint(model, c0, t, rtol=1e-5, atol=1e-6).detach().numpy()  # type: ignore


@dataclasses.dataclass(frozen=True)
class FinnDir:
    """Directory structure for a FINN experiment."""

    path: Path

    def __post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def finn_params_path(self) -> Path:
        return self.path / "finn_params.json"

    def load_finn_params(self) -> dict[str, Any]:
        return json.loads(self.finn_params_path.read_text())

    @property
    def c_train_path(self) -> Path:
        return self.path / "c_train.npy"

    @property
    def t_train_path(self) -> Path:
        return self.path / "t_train.npy"

    @property
    def u_ret_path(self) -> Path:
        return self.path / "u_ret.npy"

    @property
    def ckpt_path(self) -> Path:
        return self.path / "ckpt.pt"

    @property
    def training_vals_dir(self) -> Path:
        p = self.path / "training_vals/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def get_data_pred_path(self, epoch: int) -> Path:
        """The concentration that should equal the data (not necessarily full c field)."""
        return self.training_vals_dir / f"data_pred_{epoch}.npy"

    def get_D_eff_path(self, epoch: int) -> Path:
        return self.training_vals_dir / f"D_eff_{epoch}.npy"

    def get_cauchy_mult_path(self, epoch: int) -> Path:
        return self.training_vals_dir / f"cauchy_mult_{epoch}.npy"

    def get_p_exp_path(self, epoch: int) -> Path:
        return self.training_vals_dir / f"p_exp_{epoch}.npy"

    @property
    def retardation_predictions_dir(self) -> Path:
        p = self.path / "predicted_retardations/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def get_pred_ret_path(self, epoch: int) -> Path:
        return self.retardation_predictions_dir / f"retPred_{epoch}.npy"

    @property
    def n_epochs(self) -> int:
        return len(list(self.retardation_predictions_dir.glob("retPred*.npy")))

    @property
    def best_epoch(self) -> int:
        mses = []
        data = np.load(self.c_train_path)
        for i in range(self.n_epochs):
            pred = np.load(self.get_data_pred_path(i))
            mses.append(np.square(data - pred).mean())
        return int(np.argmin(mses))

    @property
    def best_ret_points(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.load(self.u_ret_path).reshape(-1),
            np.load(self.get_pred_ret_path(self.best_epoch)).reshape(-1),
        )

    def get_ode_pred_path(self, epoch: int) -> Path:
        return self.retardation_predictions_dir / f"c_pred_{epoch}.npy"

    @property
    def best_c_pred(self) -> np.ndarray:
        return np.load(self.get_ode_pred_path(self.best_epoch))

    @property
    def loss_path(self) -> Path:
        return self.path / "finn_loss.txt.npy"

    @property
    def processed_data_dir(self) -> Path:
        p = self.path / "processed_data/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def image_dir(self) -> Path:
        p = self.processed_data_dir / "finn_images/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def done_marker_path(self) -> Path:
        return self.path / "finn_done.marker"

    @property
    def is_done(self) -> bool:
        return self.done_marker_path.exists()


def _construct_ret_func(u_ret: np.ndarray, ret: np.ndarray):
    # TODO: interp1D_torch uses linspace but we cannot be sure that u_ret is linearly spaced
    assert np.allclose(np.diff(u_ret), np.full(len(u_ret) - 1, u_ret[1] - u_ret[0]))

    ret_tensor = torch.from_numpy(ret)
    u_min = u_ret.min()
    u_max = u_ret.max()

    def ret_fun(c):
        return interp1D_torch(ret_tensor, u_min, u_max, c)

    return ret_fun


def compute_core2_btc(
    u_ret: np.ndarray, ret: np.ndarray, cauchy_mult, D_eff
) -> np.ndarray:
    ret_fun = _construct_ret_func(u_ret, ret)

    data_core2 = load_exp_data("Core 2")
    conf_core2 = load_exp_conf("Core 2")

    t = torch.FloatTensor(data_core2["time"].to_numpy())
    finn_params = FinnParams.from_dict(is_exp_data=True, **conf_core2)
    finn_params.p_exp_flux = [0.0, 0.0]
    c0 = torch.zeros(2, finn_params.Nx, 1).to(torch.float32)
    c_ode = solve_diffusion_sorption_pde(
        retardation_fun=ret_fun, t=t, finn_params=finn_params, c0=c0
    )

    cauchy_mult = cauchy_mult * D_eff
    pred = ((c_ode[:, 0, -2] - c_ode[:, 0, -1]) * cauchy_mult).squeeze()

    return pred


def compute_core2B_profile_simple(u_ret: np.ndarray, ret: np.ndarray) -> np.ndarray:
    ret_fun = _construct_ret_func(u_ret, ret)

    data_core2b = load_exp_data("Core 2B")
    conf_core2b = load_exp_conf("Core 2B")
    time_core2b = torch.linspace(0.0, conf_core2b["T"], 101)

    t = torch.FloatTensor(time_core2b)
    finn_params = FinnParams.from_dict(is_exp_data=True, **conf_core2b)
    finn_params.p_exp_flux = [0.0, 0.0]
    c0 = torch.zeros(2, finn_params.Nx, 1).to(torch.float32)
    c_ode = solve_diffusion_sorption_pde(
        retardation_fun=ret_fun, t=t, finn_params=finn_params, c0=c0
    )

    x = data_core2b["x"].to_numpy()
    xp = np.linspace(0.0, conf_core2b["X"], int(conf_core2b["Nx"]))
    profile = np.interp(x, xp, c_ode[-1, 1, :, 0])

    return profile


def compute_core2B_profile(finn_dir: FinnDir, u_and_ret=None) -> np.ndarray:
    if u_and_ret is None:
        u_ret, ret = finn_dir.best_ret_points
    else:
        u_ret, ret = u_and_ret
    return compute_core2B_profile_simple(u_ret, ret)


class Training:
    def __init__(
        self,
        model,
        finn_dir: FinnDir,
        continue_training: bool = False,
        n_epochs: int = 30,
        error_mult: float = 1e5,
        phys_mult: float = 1e2,
        start_lr: float = 1e-1,
        use_adam_optim: bool = False,
    ):
        self.model = model
        self.finn_dir = finn_dir
        self.error_mult = error_mult
        self.phys_mult = phys_mult
        self.start_lr = start_lr
        self.n_epochs = n_epochs

        if use_adam_optim:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
        else:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=start_lr)  # type: ignore

        self.start_epoch = 0
        self.train_losses = []
        self.best_loss = np.infty

        self.latest_mse_loss = None
        self.latest_physical_loss = None
        self.latest_pred = None
        self.latest_ode_pred = None
        self.latest_D_eff = None
        self.latest_cauchy_mult = None

        # Load the model from checkpoint
        if continue_training:
            print("Restoring model (that is the network's weights) from file...\n")

            self.checkpoint = torch.load(self.finn_dir.ckpt_path)

            # Load the model state_dict (all the network parameters)
            self.model.load_state_dict(self.checkpoint["state_dict"])

            # Load the optimizer state dict (important because ADAM and LBFGS
            # requires past states, e.g. momentum information and approximate
            # Hessian)
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v

            # Load the epoch and loss values from the previous training up until
            # the checkpoint to enable complete history of the training
            self.start_epoch = self.checkpoint["epoch"]
            self.train_losses = self.checkpoint["loss_train"]

    def model_train(
        self, *, u0: torch.Tensor, t: torch.Tensor, data: torch.Tensor
    ) -> None:
        finn_dir = self.finn_dir
        u = torch.linspace(0.0, 2.0, 100).view(-1, 1)  # TODO: Adjust limits and n
        np.save(finn_dir.u_ret_path, u)

        criterion = torch.nn.MSELoss()

        def closure():
            # Set the model to training mode
            self.model.train()

            # Reset the gradient buffer (set to 0)
            self.optimizer.zero_grad()

            # Calculate the model prediction (full field solution)
            ode_pred: torch.Tensor = odeint(self.model, u0, t, rtol=1e-5, atol=1e-6)  # type: ignore
            self.latest_ode_pred = ode_pred.clone().detach().numpy()

            # Extract the breakthrough curve from the full field solution prediction
            cauchy_mult = (
                self.model.flux_modules[0].cauchy_mult
                * self.model.flux_modules[0].D_eff
            )
            pred = ((ode_pred[:, 0, -2] - ode_pred[:, 0, -1]) * cauchy_mult).squeeze()

            loss = self.error_mult * criterion(data, pred)

            # Extract the predicted retardation factor function for physical
            # regularization
            ret_temp = self.model.flux_modules[0].coeff_nn(u)
            # normalize retardation such that there is no difference whether the rets are orders of magnitude apart but just if the slope is different
            ret_temp = ret_temp / ret_temp.max()

            # Physical regularization: value of the retardation factor should
            # decrease with increasing concentration
            loss_physical = (
                1000
                * self.phys_mult
                * torch.mean(torch.relu(ret_temp[:-1] - ret_temp[1:]))
            )
            self.latest_mse_loss = loss
            self.latest_physical_loss = loss_physical

            loss += loss_physical

            loss.backward()

            self.latest_pred = pred.clone().detach().numpy()
            self.latest_D_eff = self.model.flux_modules[0].D_eff
            self.latest_cauchy_mult = self.model.flux_modules[0].cauchy_mult

            return loss

        # learning_rates = (
        #     # np.exp(-1.75 * np.linspace(0, 1, self.n_epochs)) * self.start_lr
        #     np.exp(-2.5 * np.linspace(0, 1, self.n_epochs)) * self.start_lr
        # )
        lr_min, lr_max, decay_factor, T_0 = 1e-2, 0.1, 0.8, 10
        num_restarts = 30

        learning_rates = np.concatenate(
            [
                lr_min
                + 0.5
                * (lr_max * decay_factor**i - lr_min)
                * (1 + np.cos(np.linspace(0, np.pi, T_0)))
                for i in range(num_restarts)
            ]
        )

        # Iterate until maximum epoch number is reached
        for epoch in range(self.start_epoch, self.n_epochs):
            for g in self.optimizer.param_groups:
                g["lr"] = learning_rates[epoch]

            dt = time.time()

            # Update the model parameters and record the loss value
            loss = self.optimizer.step(closure)
            self.train_losses.append(loss.item())  # type: ignore

            dt = time.time() - dt

            print(
                f"It {epoch+1:>3}/{self.n_epochs}"
                f" | mse = {np.square(self.latest_pred - data.detach().numpy()).mean():.2e}"
                f" | loss = {self.train_losses[-1]:.2e}"
                f" | dt = {dt:.1f}s"
                f" | lr = {learning_rates[epoch]:.1e}"
                f" | loss_mse = {self.latest_mse_loss:.1e}"
                f" | loss_phys = {self.latest_physical_loss:.1e}"
            )

            ret_pred = (
                1
                / self.model.flux_modules[0].coeff_nn(u)
                / 10 ** self.model.flux_modules[0].p_exp
            )

            fig, axs = plt.subplots(ncols=2, figsize=(12, 3))
            axs[0].plot(data.detach().numpy(), "r.", alpha=0.3)
            axs[0].plot(self.latest_pred, "b-")

            axs[1].plot(
                u.detach().numpy(), ret_pred.detach().numpy(), "b-", label="Prediction"
            )
            axs[1].plot(
                u.detach().numpy(),
                np.load(
                    "/Users/r/Documents/dev/tmp/finn_with_julia/python/diffusion_sorption/experimental_data/learned_retardation.npy"
                ),
                "k--",
                label="I need this",
            )
            axs[1].legend()
            fig.savefig(finn_dir.image_dir / f"learned_data_{epoch}.png")
            # plt.show()

            assert self.latest_ode_pred is not None
            assert self.latest_pred is not None
            assert self.latest_D_eff is not None
            assert self.latest_cauchy_mult is not None
            np.save(finn_dir.get_pred_ret_path(epoch), ret_pred.detach().numpy())
            np.save(finn_dir.get_data_pred_path(epoch), self.latest_pred)
            np.save(finn_dir.get_ode_pred_path(epoch), self.latest_ode_pred)
            np.save(finn_dir.get_D_eff_path(epoch), self.latest_D_eff)
            np.save(finn_dir.get_cauchy_mult_path(epoch), self.latest_cauchy_mult)
            np.save(
                finn_dir.get_p_exp_path(epoch),
                self.model.flux_modules[0].p_exp.detach().numpy(),
            )

            if (epoch + 1) % 5 == 0:
                self.save_model_to_file(epoch)

        np.save(finn_dir.loss_path, self.train_losses)

    def save_model_to_file(self, epoch):
        state = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_train": self.train_losses,
        }
        torch.save(state, self.finn_dir.ckpt_path)


@dataclasses.dataclass(frozen=True)
class P3innDir:
    """Directory structure for a P3INN experiment."""

    path: Path

    def __post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def p3inn_params_path(self) -> Path:
        # TODO: write this in p3inn_compute function
        return self.path / "p3inn_params.json"

    @property
    def x_data_path(self) -> Path:
        return self.path / "x_data.npy"

    @property
    def y_data_path(self) -> Path:
        return self.path / "y_data.npy"

    @property
    def loss_path(self) -> Path:
        # TODO: write this in p3inn_compute function
        return self.path / "p3inn_loss.txt"

    @property
    def x_eval_path(self) -> Path:
        return self.path / "x_eval.npy"

    @property
    def pred_mean_path(self) -> Path:
        return self.path / "pred_mean.npy"

    @property
    def pred_median_path(self) -> Path:
        return self.path / "pred_median.npy"

    @property
    def pred_PIs_dir(self) -> Path:
        p = self.path / "pred_PIs/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def get_pred_bound_path(
        self, quantile: float, bound_type: Literal["up", "down"]
    ) -> Path:
        return self.pred_PIs_dir / f"PI_{bound_type}_{quantile:g}.npy"

    def iter_pred_PIs(self):
        quantile_and_PI_paths = []
        for up_path in self.pred_PIs_dir.glob("*_up_*.npy"):
            quantile = float(up_path.stem.split("_")[-1])
            down_path = up_path.parent / up_path.name.replace("_up_", "_down_")
            quantile_and_PI_paths.append((quantile, up_path, down_path))

        quantile_and_PI_paths = sorted(quantile_and_PI_paths, key=lambda t: t[0])

        for q, u, d in quantile_and_PI_paths:
            yield q, np.load(u), np.load(d)

    @property
    def processed_data_dir(self) -> Path:
        p = self.path / "processed_data/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def image_dir(self) -> Path:
        p = self.processed_data_dir / "p3inn_images/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def done_marker_path(self) -> Path:
        return self.path / "p3inn_done.marker"

    @property
    def is_done(self) -> bool:
        return self.done_marker_path.exists()


def solve_diffusion_sorption_pde_bash(
    out_path, retardation_factors_path: str, c_diss_max: float, pde_params, c0_path=None
):
    """
    Solve a diffusion-sorption PDE using the provided boundary conditions and other parameters specified in `pde_params` and return the solution.

    Approximate the retardation function R(c) lineary using equally spaced points from 0 to `c_diss_max` (inclusive) and the retardation factor values located at the given file path.

    Args:
        retardation_factors_path (Path): _description_
        c_diss_max (float): Largest interpolation value for R(c)
        pde_params (dict): PDE parameters like domain size and physical constants.
        c0_path (Path, optional): Path to initial conditions. Defaults to zero.

    Returns:
        np.ndarray: A numpy array of shape (Nt, 2, Nx) representing the
        solution c(x,t), where `Nt` is the number of time steps, and
        `Nx` is the number of spatial points.
    """

    pass


def finn_fit_retardation(
    out_dir, is_exp_data: bool, c_train_path=None, t_train_path=None, **finn_dict
):
    finn_dir = FinnDir(Path(out_dir))
    if finn_dir.is_done:
        return

    finn_params = FinnParams.from_dict(is_exp_data=is_exp_data, **finn_dict)
    finn_dir.finn_params_path.write_text(json.dumps(finn_params.to_dict()))

    if c_train_path is None:
        c_train_path = finn_dir.c_train_path
    if t_train_path is None:
        t_train_path = finn_dir.t_train_path

    t_train = torch.from_numpy(np.load(t_train_path)).to(torch.float32)
    c_train = torch.from_numpy(np.load(c_train_path)).to(torch.float32)
    np.save(finn_dir.t_train_path, t_train.numpy())
    np.save(finn_dir.c_train_path, c_train.numpy())
    assert c_train.shape[0] == t_train.shape[0]

    Nx = finn_params.Nx
    Ny = 1
    if c_train.ndim == 4:  # synthetic data case: c_train has shape (Nt, 2, Nx, Ny)
        c0 = c_train[0].clone()
    else:
        # FIXME: Implement this as above
        # core2 : c_train has shape (Nt,)
        # core2B: c_train has shape (Nx,)
        c0 = torch.zeros(2, Nx, Ny)

    model = Net_Model(c0, finn_params, ret_funs=[make_mlp(), None])
    trainer = Training(
        model,
        finn_dir=finn_dir,
        n_epochs=finn_params.n_epochs,
        error_mult=finn_params.error_mult,
        phys_mult=finn_params.phys_mult,
        start_lr=finn_params.start_lr,
        use_adam_optim=finn_params.use_adam_optim,
    )
    trainer.model_train(u0=c0, t=t_train, data=c_train)

    finn_dir.done_marker_path.touch()


def pi3nn_compute_PI_and_mean(
    out_dir, quantiles: list[float], x_data_path=None, y_data_path=None, seed=None
):
    p3inn_dir = P3innDir(out_dir)
    if p3inn_dir.is_done:
        return

    if x_data_path is None:
        x_data_path = p3inn_dir.x_data_path
    if y_data_path is None:
        y_data_path = p3inn_dir.y_data_path

    X = np.load(x_data_path)
    Y = np.load(y_data_path)
    np.save(p3inn_dir.x_data_path, X)
    np.save(p3inn_dir.y_data_path, Y)

    plt.plot(X, Y, ".")
    plt.savefig(p3inn_dir.image_dir / "input_data.png")
    # plt.show()

    SEED = int(hash(out_dir)) % 2**31 if seed is None else int(seed)

    # random split
    xTrainValid, xTest, yTrainValid, yTest = train_test_split(
        X, Y, test_size=0.1, random_state=SEED, shuffle=True
    )
    ## Split the validation data
    xTrain, xValid, yTrain, yValid = train_test_split(
        xTrainValid, yTrainValid, test_size=0.1, random_state=SEED, shuffle=True
    )

    ### Data normalization
    scalar_x = StandardScaler()
    scalar_y = StandardScaler()

    xTrain = scalar_x.fit_transform(xTrain)
    xValid = scalar_x.transform(xValid)
    xTest = scalar_x.transform(xTest)

    yTrain = scalar_y.fit_transform(yTrain)
    yValid = scalar_y.transform(yValid)
    yTest = scalar_y.transform(yTest)

    ### To tensors
    xTrain = torch.Tensor(xTrain)
    xValid = torch.Tensor(xValid)
    xTest = torch.Tensor(xTest)

    yTrain = torch.Tensor(yTrain)
    yValid = torch.Tensor(yValid)
    yTest = torch.Tensor(yTest)

    plt.plot(xTrain, yTrain, ".")
    plt.plot(xTest, yTest, "x")
    # plt.plot(xValid, yValid, "d")
    plt.savefig(p3inn_dir.image_dir / "transformed_data_split.png")
    # plt.show()

    #########################################################
    ############## End of Data Loading Section ##############
    #########################################################
    num_inputs = 1
    num_outputs = 1

    configs = {
        "seed": SEED,
        "num_neurons_mean": [64],
        "num_neurons_up": [64],
        "num_neurons_down": [64],
        "Max_iter": 50000,
    }
    import random

    LR = 1e-2
    DECAY_RATE = 0.96  # Define decay rate
    DECAY_STEPS = 10000  # Define decay steps
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])

    """ Create network instances"""
    net_mean = UQ_Net_mean(configs, num_inputs, num_outputs)
    net_up = UQ_Net_std(configs, num_inputs, num_outputs, net="up")
    net_down = UQ_Net_std(configs, num_inputs, num_outputs, net="down")

    # Initialize trainer and conduct training/optimizations
    trainer = CL_trainer(
        configs,
        net_mean,
        net_up,
        net_down,
        x_train=xTrain,
        y_train=yTrain,
        x_valid=xValid,
        y_valid=yValid,
        x_test=xTest,
        y_test=yTest,
        lr=LR,
        decay_rate=DECAY_RATE,  # Pass decay rate to trainer
        decay_steps=DECAY_STEPS,  # Pass decay steps to trainer
    )
    trainer.train()  # training for 3 networks

    full_preds = trainer.eval_networks(
        torch.from_numpy(scalar_x.transform(X)).to(torch.float32)
    )
    y_mean_full = scalar_y.inverse_transform(full_preds["mean"])
    y_median_full = shift_to_median(y_mean_full, Y)

    median_shift = compute_shift_to_median(y_mean_full, Y)

    print(median_shift)
    print((y_median_full - Y > 0).sum())
    print((y_median_full - Y < 0).sum())

    x_curve = torch.linspace(X.min(), X.max(), 100, dtype=torch.float32).reshape(-1, 1)
    x_curve = torch.from_numpy(scalar_x.transform(x_curve)).to(torch.float32)
    pred_curves = trainer.eval_networks(x_curve)
    pred_curves["median"] = pred_curves["mean"] + median_shift
    x_curve = x_curve.detach().numpy()
    x_curve = scalar_x.inverse_transform(x_curve)

    for quantile in quantiles:
        c_up, c_down = compute_boundary_factors(
            y_train=yTrain.numpy(),
            network_preds=trainer.eval_networks(xTrain, as_numpy=True),
            quantile=quantile,
            verbose=1,
        )

        assert c_up > 0 and c_down > 0
        y_U_PI_array_train = (pred_curves["median"] + c_up * pred_curves["up"]).numpy()
        y_L_PI_array_train = (
            pred_curves["median"] - c_down * pred_curves["down"]
        ).numpy()
        y_mean = pred_curves["mean"].numpy()
        y_median = pred_curves["median"].numpy()

        y_mean = scalar_y.inverse_transform(y_mean)
        y_median = scalar_y.inverse_transform(y_median)
        y_U_PI_array_train = scalar_y.inverse_transform(y_U_PI_array_train)
        y_L_PI_array_train = scalar_y.inverse_transform(y_L_PI_array_train)

        assert torch.all(pred_curves["up"] > 0)
        assert torch.all(pred_curves["down"] > 0)

        # since c > 0 and std network prediction > 0, the median should always be between the upper and lower prediction intervals and not cross them
        assert np.all(y_median > y_L_PI_array_train)
        assert np.all(y_median < y_U_PI_array_train)

        # Scatter plot of the original data points
        np.save(p3inn_dir.x_eval_path, x_curve)
        np.save(p3inn_dir.pred_mean_path, y_mean)
        np.save(p3inn_dir.pred_median_path, y_median)
        np.save(p3inn_dir.get_pred_bound_path(quantile, "up"), y_U_PI_array_train)
        np.save(p3inn_dir.get_pred_bound_path(quantile, "down"), y_L_PI_array_train)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Scatter plot of the original data points
    ax1.plot(X, Y, ".", alpha=0.2)

    # Plot the mean curve
    # ax1.plot(x_curve, y_mean, "b-", label="Mean Prediction")
    ax1.plot(x_curve, y_median, "b-", label="Median Prediction")

    # Fill the area between the upper and lower prediction intervals with transparency
    ax1.fill_between(
        x_curve.flatten(),
        y_L_PI_array_train.flatten(),
        y_U_PI_array_train.flatten(),
        color="grey",
        alpha=0.3,
        label="Prediction Interval",
    )
    for i, (q, bound_up, bound_down) in enumerate(p3inn_dir.iter_pred_PIs()):
        ax2.fill_between(
            x=x_curve.flat,
            y1=bound_down.flat,
            y2=bound_up.flat,
            color=f"C{0}",
            alpha=0.2,
        )
    ax2.plot(X, Y, ".")
    ax1.legend()
    fig.savefig(p3inn_dir.image_dir / "learned_data.png")

    # Plot residual training results
    data_train_up, data_train_down = create_PI_training_data(
        trainer.networks["mean"], X=trainer.x_train, Y=trainer.y_train
    )

    ## up
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
    ax1.plot(*data_train_up, ".")
    ax1.plot(
        xTrain.detach().numpy(),
        trainer.networks["up"](xTrain).detach().numpy(),
        "x",
        alpha=0.2,
    )
    ax2.plot(
        xTrain.detach().numpy(), trainer.networks["up"](xTrain).detach().numpy(), "."
    )
    fig.savefig(p3inn_dir.image_dir / "learned_residual_up.png")

    ## down
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
    ax1.plot(*data_train_down, ".")
    ax1.plot(
        xTrain.detach().numpy(),
        trainer.networks["down"](xTrain).detach().numpy(),
        "x",
        alpha=0.2,
    )
    ax2.plot(
        xTrain.detach().numpy(), trainer.networks["down"](xTrain).detach().numpy(), "."
    )
    fig.savefig(p3inn_dir.image_dir / "learned_residual_down.png")
    # plt.show()

    p3inn_dir.done_marker_path.touch()


def _load_exp_df(name: Literal["Core 1", "Core 2", "Core 2B"], sheet) -> pd.DataFrame:
    p = Path(f"../data/experimental_data/data_{name.replace(' ', '').lower()}.xlsx")
    return pd.read_excel(p, index_col=None, header=None, sheet_name=sheet)


def load_exp_conf(name: Literal["Core 1", "Core 2", "Core 2B"]) -> dict[str, Any]:
    """
    0	D	0.000020	m^2/day
    1	por	0.288000	-
    2	rho_s	1957.000000	kg/m^3
    4	X	0.026035	m
    5	T	39.824440	days
    6	Nx	20.000000	-
    7	Nt	55.000000	-
    8	sample_radius	0.023750	m
    9	Q	0.000104	m^3/day
    11	solubility	1.600000	kg/m^3
    12	Dirichlet	0.000000	NaN
    13	Cauchy	1.000000	NaN
    """
    df = _load_exp_df(name, sheet=1).dropna(how="all")
    # df.columns = ["label", "value", "unit"]
    return df[[0, 1]].set_index(0, drop=True).to_dict()[1]  # type: ignore
    # return df


def load_exp_data(name: Literal["Core 1", "Core 2", "Core 2B"], physical_model=False):
    df = _load_exp_df(name, sheet=2 if physical_model else 0)
    if name == "Core 2B":
        df.columns = ["x", "c_tot"]
        df["c_tot"] /= 1000.0
    else:
        df.columns = ["time", "c_diss"]
        df["c_diss"] /= 1000.0
    return df
