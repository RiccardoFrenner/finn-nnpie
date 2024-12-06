import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint

import plotting


@dataclasses.dataclass
class ExperimentalSamples:
    core1: np.ndarray
    core2: np.ndarray
    core2b: np.ndarray
    ret_x: np.ndarray
    ret_y: np.ndarray

    @classmethod
    def from_dir(cls, p):
        def try_load(p):
            try:
                return np.load(p)
            except FileNotFoundError:
                return np.array([[]])

        p = Path(p).resolve()
        return cls(
            core1=try_load(p / "y_core1_samples.npy"),
            core2=try_load(p / "y_core2_samples.npy"),
            core2b=try_load(p / "y_core2B_samples.npy"),
            ret_x=try_load(p / "x_ret_samples.npy").squeeze(),
            ret_y=try_load(p / "y_ret_samples.npy"),
        )

    def to_dir(self, p):
        p = Path(p).resolve()
        np.save(p / "y_core1_samples.npy", self.core1)
        np.save(p / "y_core2_samples.npy", self.core2)
        np.save(p / "y_core2B_samples.npy", self.core2b)
        np.save(p / "x_ret_samples.npy", self.ret_x)
        np.save(p / "y_ret_samples.npy", self.ret_y)

    def plot(
        self,
        axs: Optional[list[plt.Axes]] = None,
        set_titles: bool = True,
        line_kwargs=None,
        only_outlines: bool = False,
    ):
        if axs is None:
            fig, axs = plt.subplots(
                ncols=2,
                nrows=2,
                figsize=(2 * plotting.FIGURE_WIDTH, 2 * plotting.FIGURE_HEIGHT),
            )
            axs = axs.flatten().tolist()
        else:
            fig = plt.gcf()
        assert axs is not None
        line_kwargs = line_kwargs or dict()

        if set_titles:
            axs[0].set_title("Core 1")
            axs[1].set_title("Core 2")
            axs[2].set_title("Core 2B")
            axs[3].set_title("Retardation")

        core1_x = load_exp_data_numpy("Core 1")[0]
        core2_x = load_exp_data_numpy("Core 2")[0]
        core2B_conf = load_exp_conf("Core 2B")
        if self.core2b.shape[1] == int(core2B_conf["Nx"]):
            core2b_x = np.linspace(0, core2B_conf["X"], int(core2B_conf["Nx"]))
        else:
            core2b_x = load_exp_data_numpy("Core 2B")[0]

        line_kwargs.setdefault("color", "black")
        line_kwargs.setdefault("linestyle", "-")

        def compute_outlines(arr):
            return np.array(
                [
                    np.min(arr, axis=0),
                    np.max(arr, axis=0),
                ]
            )

        if only_outlines:
            line_kwargs.setdefault("alpha", 0.5)
            if self.core1.size > 0:
                axs[0].fill_between(
                    core1_x, *compute_outlines(self.core1), **line_kwargs
                )
            if self.core2.size > 0:
                axs[1].fill_between(
                    core2_x, *compute_outlines(self.core2), **line_kwargs
                )
            if self.core2b.size > 0:
                axs[2].fill_between(
                    core2b_x, *compute_outlines(self.core2b), **line_kwargs
                )
            if self.ret_x.size > 0:
                axs[3].fill_between(
                    self.ret_x, *compute_outlines(self.ret_y), **line_kwargs
                )
        else:
            line_kwargs["alpha"] = max(1e-2, min(1.0, 6 / self.core2.shape[0]))
            if self.core1.size > 0:
                axs[0].plot(core1_x, self.core1.T, **line_kwargs)
            if self.core2.size > 0:
                axs[1].plot(core2_x, self.core2.T, **line_kwargs)
            if self.core2b.size > 0:
                axs[2].plot(core2b_x, self.core2b.T, **line_kwargs)
            if self.ret_x.size > 0:
                axs[3].plot(self.ret_x, self.ret_y.T, **line_kwargs)

        axs[0].sharex(axs[1])
        axs[1].set_yticklabels([])
        # axs[0].set_ylabel(plotting.C_DISS_Y_LABEL)
        # axs[2].set_ylabel(plotting.C_TOT_Y_LABEL)

        axs[3].set_xlim(0, 1.5)

        plotting.set_retardation_axes_stuff(axs[-1], set_xlabel=True, set_ylabel=True)
        for i, ax in enumerate(axs[:-1]):
            plotting.set_concentration_axes_stuff(
                ax,
                core="2" if i != 2 else "2B",
                set_xlabel=i in [1, 2],
                set_ylabel=i in [0, 2],
            )

        return fig, axs


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
    def core1_btc(self) -> Path:
        return self.path / "core1_btc.npy"

    @property
    def core2b_profile(self) -> Path:
        return self.path / "core2b_profile.npy"

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


def compute_btc(
    u_ret: np.ndarray,
    ret: np.ndarray,
    cauchy_mult,
    D_eff,
    core_type: Literal["Core 1", "Core 2"],
) -> np.ndarray:
    ret_fun = _construct_ret_func(u_ret, ret)

    data = load_exp_data(core_type)
    conf = load_exp_conf(core_type)

    t = torch.FloatTensor(data["time"].to_numpy())
    finn_params = FinnParams.from_dict(is_exp_data=True, **conf)
    finn_params.p_exp_flux = [0.0, 0.0]
    c0 = torch.zeros(2, finn_params.Nx, 1).to(torch.float32)
    c_ode = solve_diffusion_sorption_pde(
        retardation_fun=ret_fun, t=t, finn_params=finn_params, c0=c0
    )

    if core_type == "Core 1":
        cauchy_mult = 0.0836712021582612
    else:
        cauchy_mult = cauchy_mult * D_eff

    pred = ((c_ode[:, 0, -2] - c_ode[:, 0, -1]) * cauchy_mult).squeeze()

    return pred


def compute_core1_btc(u_ret: np.ndarray, ret: np.ndarray) -> np.ndarray:
    finn_dir = FinnDir(Path("../data_out/finn/core2").resolve())
    finn_params = finn_dir.load_finn_params()
    cauchy_mult = finn_params["cauchy_mult"][0]
    D_eff = finn_params["D_eff"][0]
    return compute_btc(u_ret, ret, cauchy_mult, D_eff, "Core 1")


def compute_core2_btc(
    u_ret: np.ndarray, ret: np.ndarray, cauchy_mult=None, D_eff=None
) -> np.ndarray:
    finn_dir = FinnDir(Path("../data_out/finn/core2").resolve())
    finn_params = finn_dir.load_finn_params()
    if cauchy_mult is None:
        cauchy_mult = finn_params["cauchy_mult"][0]
    if D_eff is None:
        D_eff = finn_params["D_eff"][0]
    return compute_btc(u_ret, ret, cauchy_mult, D_eff, "Core 2")


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
    write_to_file = False
    if u_and_ret is None:
        u_ret, ret = finn_dir.best_ret_points
        if finn_dir.core2b_profile.exists():
            return np.load(finn_dir.core2b_profile)
        else:
            write_to_file = True
    else:
        u_ret, ret = u_and_ret
    profile = compute_core2B_profile_simple(u_ret, ret)
    if write_to_file:
        np.save(finn_dir.core2b_profile, profile)
    return profile


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


def load_exp_data_numpy(
    name: Literal["Core 1", "Core 2", "Core 2B"], physical_model=False
):
    return load_exp_data(name, physical_model).to_numpy().T
