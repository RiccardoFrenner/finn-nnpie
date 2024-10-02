from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class NetworkHyperparameters:
    nodes_per_layer: List[int] = field(default_factory=lambda: [16, 32, 16])
    error_mult: int = 26 * 51  # multiplier for squared error
    phys_mult: int = 100 * 26 * 51  # multiplier for physical regularization


@dataclass
class SoilParameters:
    D: float = 0.0005  # effective diffusion coefficient [m^2/day]
    por: float = 0.29  # porosity [-]
    rho_s: float = 2880  # bulk density [kg/m^3]
    Kf: float = field(init=False)  # freundlich's K [(m^3/kg)^nf]
    nf: float = 0.874  # freundlich exponent [-]
    smax: float = 1 / 1700  # sorption capacity [m^3/kg]
    Kl: float = 1  # half-concentration [kg/m^3]
    Kd: float = 0.429 / 1000  # organic carbon partitioning [m^3/kg]
    solubility: float = 1.0  # top boundary value [kg/m^3]

    def __post_init__(self):
        self.Kf = 1.016 / self.rho_s


@dataclass
class SimulationDomain:
    X: float = 1.0  # length of sample [m]
    _dx: float = 0.04  # length of discrete control volume [m]
    T: float = 10000  # simulation time [days]
    _dt: float = 5  # time step [days]
    cauchy_val: float = field(init=False)

    def __post_init__(self):
        self.update_cauchy_val()

    @property
    def Nx(self) -> int:
        return int(self.X / self._dx) + 1

    @Nx.setter
    def Nx(self, value: int):
        self._dx = self.X / (value - 1)

    @property
    def dx(self) -> float:
        return self._dx

    @dx.setter
    def dx(self, value: float):
        self._dx = value
        self.update_cauchy_val()

    @property
    def Nt(self) -> int:
        return int(self.T / self._dt) + 1

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float):
        self._dt = value

    def update_cauchy_val(self):
        # Assuming some predefined parameters
        r = 0.1  # Example sample radius
        A = np.pi * r**2
        Q = 1.0  # Example flow rate
        self.cauchy_val = self._dx * self.X * A / Q


@dataclass
class BoundaryConditions:
    dirichlet_bool: List[List[bool]] = field(
        default_factory=lambda: [
            [True, False, False, False],
            [True, False, False, False],
        ]
    )
    neumann_bool: List[List[bool]] = field(
        default_factory=lambda: [[False, False, True, True], [False, False, True, True]]
    )
    cauchy_bool: List[List[bool]] = field(
        default_factory=lambda: [
            [False, True, False, False],
            [False, True, False, False],
        ]
    )
    dirichlet_val: List[List[float]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    neumann_val: List[List[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    cauchy_mult: List[float] = field(default_factory=lambda: [0.04, 0.04])


@dataclass
class Parameters:
    network: NetworkHyperparameters = NetworkHyperparameters()
    soil: SoilParameters = SoilParameters()
    domain: SimulationDomain = SimulationDomain()
    boundary: BoundaryConditions = BoundaryConditions()
    learn_stencil: List[bool] = field(default_factory=lambda: [False, False])
    D_eff: List[float] = field(default_factory=lambda: [0.0005 / 0.04**2, 0.25])
    learn_coeff: List[bool] = field(default_factory=lambda: [False, True])
    is_retardation_a_func: List[bool] = field(default_factory=lambda: [True, False])

    @classmethod
    def from_excel(
        cls, filename: str, params: Optional["Parameters"] = None
    ) -> "Parameters":
        if params is None:
            params = cls()

        in_params = pd.read_excel(filename, sheet_name=1, index_col=0, header=None)

        # Update soil parameters
        params.soil.D = in_params[1].get("D", params.soil.D)
        params.soil.por = in_params[1].get("por", params.soil.por)
        params.soil.rho_s = in_params[1].get("rho_s", params.soil.rho_s)
        params.soil.solubility = in_params[1].get("solubility", params.soil.solubility)

        # Update simulation domain
        params.domain.X = in_params[1].get("X", params.domain.X)
        params.domain.Nx = int(in_params[1].get("Nx", params.domain.Nx))
        params.domain.T = in_params[1].get("T", params.domain.T)
        r = in_params[1].get("sample_radius", 0.1)  # default radius if not provided
        A = np.pi * r**2
        Q = in_params[1].get("Q", 1.0)
        params.domain.cauchy_val = params.soil.por * A / Q * params.domain.dx

        # Update diffusion coefficients
        params.D_eff = [
            params.soil.D / (params.domain.dx**2),
            params.soil.D
            * params.soil.por
            / (params.soil.rho_s / 1000)
            / (params.domain.dx**2),
        ]

        # Update boundary conditions
        params.boundary.dirichlet_bool[0][1] = bool(
            in_params[1].get("Dirichlet", False)
        )
        params.boundary.cauchy_bool[0][1] = bool(in_params[1].get("Cauchy", False))
        params.boundary.dirichlet_val[0][0] = params.soil.solubility
        params.boundary.cauchy_mult[0] = params.domain.cauchy_val

        return params
