from typing import Optional

import numpy as np
import pandas as pd


class Parameters:
    def __init__(self):
        # NETWORK HYPER-PARAMETERS
        self.flux_layers = 3  # number of hidden layers for the NN in the flux kernels
        self.state_layers = 3  # number of hidden layers for the NN in the state kernels
        self.flux_nodes = (
            15  # number of hidden nodes per layer for the NN in the flux kernels
        )
        self.state_nodes = (
            15  # number of hidden nodes per layer for the NN in the flux kernels
        )
        self.error_mult = (
            26 * 51
        )  # multiplier for the squared error in the loss function
        self.phys_mult = (
            100 * 26 * 51
        )  # multiplier for the physical regularization in the loss function

        self.lbfgs_optim = True  # Use L-BFGS as optimizer, else use ADAM
        self.train_breakthrough = False  # Train using only breakthrough curve data
        self.linear = False  # Training data generated with the linear isotherm
        self.freundlich = True  # Training data generated with the freundlich isotherm
        self.langmuir = False  # Training data generated with the langmuir isotherm

        # SIMULATION-RELATED INPUTS
        self.num_vars = 2

        # Soil Parameters
        self.D = 0.0005  # effective diffusion coefficient [m^2/day]
        self.por = 0.29  # porosity [-]
        self.rho_s = 2880  # bulk density [kg/m^3]
        self.Kf = 1.016 / self.rho_s  # freundlich's K [(m^3/kg)^nf]
        self.nf = 0.874  # freundlich exponent [-]
        self.smax = 1 / 1700  # sorption capacity [m^3/kg]
        self.Kl = 1  # half-concentration [kg/m^3]
        self.Kd = 0.429 / 1000  # organic carbon partitioning [m^3/kg]
        self.solubility = 1.0  # top boundary value [kg/m^3]

        # Simulation Domain
        self.X = 1.0  # length of sample [m]
        self.dx = 0.04  # length of discrete control volume [m]
        self.T = 10000  # simulation time [days]
        self.dt = 5  # time step [days]
        self.Nx = int(self.X / self.dx + 1)
        self.Nt = int(self.T / self.dt + 1)
        self.cauchy_val = self.dx

        # Inputs for Flux Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_flux = [self.flux_layers, self.flux_layers]
        self.num_nodes_flux = [self.flux_nodes, self.flux_nodes]
        ## Set numerical stencil to be learnable or not
        self.learn_stencil = [False, False]
        ## Effective diffusion coefficient for each variable D_eff = [D / (dx**2), D * por / (rho_s/1000) / (dx**2)]
        self.D_eff = [self.D / (self.dx**2), 0.25]
        ## Set diffusion coefficient to be learnable or not
        self.learn_coeff = [False, True]
        ## Set if diffusion coefficient to be approximated as a function
        self.is_retardation_a_func = [True, False]
        ## Normalizer for functions that are approximated with a NN
        self.p_exp_flux = [0.0, 0.0]
        ## Set the variable index to be used when calculating the fluxes
        self.flux_calc_idx = [0, 0]
        ## Set the variable indices necessary to calculate the diffusion
        ## coefficient function
        self.flux_couple_idx = [0, 0]
        ## Set boundary condition types
        self.dirichlet_bool = [[True, False, False, False], [True, False, False, False]]
        self.neumann_bool = [[False, False, True, True], [False, False, True, True]]
        self.cauchy_bool = [[False, True, False, False], [False, True, False, False]]
        ## Set the Dirichlet and Neumann boundary values if necessary, otherwise set = 0
        self.dirichlet_val = [
            [self.solubility, 0.0, 0.0, 0.0],
            [self.solubility, 0.0, 0.0, 0.0],
        ]
        self.neumann_val = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ## Set multiplier for the Cauchy boundary condition if necessary
        ## (will be multiplied with D_eff in the flux kernels), otherwise self.set = 0
        self.cauchy_mult = [self.dx, self.dx]

        # Inputs for State Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_state = [self.state_layers, self.state_layers]
        self.num_nodes_state = [self.state_nodes, self.state_nodes]
        ## Normalizer for the reaction functions that are approximated with a NN
        self.p_exp_state = [0.0, 0.0]
        ## Set the variable indices necessary to calculate the reaction function
        self.state_couple_idx = [0, 1]

    @classmethod
    def from_excel(cls, filename, params: Optional["Parameters"] = None):
        if params is None:
            params = cls()

        # Load parameters from the Excel file
        in_params = pd.read_excel(filename, sheet_name=1, index_col=0, header=None)

        # Soil Parameters
        params.D = in_params[1]["D"]
        params.por = in_params[1]["por"]
        params.rho_s = in_params[1]["rho_s"]

        # Simulation Domain
        params.X = in_params[1]["X"]
        params.Nx = int(in_params[1]["Nx"])
        params.dx = params.X / (params.Nx + 1)
        params.T = in_params[1]["T"]
        r = in_params[1]["sample_radius"]
        A = np.pi * r**2
        Q = in_params[1]["Q"]
        params.solubility = in_params[1]["solubility"]
        params.cauchy_val = params.por * A / Q * params.dx

        # Inputs for Flux Kernels
        ## Effective diffusion coefficient for each variable
        params.D_eff = [
            params.D / (params.dx**2),
            params.D * params.por / (params.rho_s / 1000) / (params.dx**2),
        ]
        ## Set diffusion coefficient to be learnable or not
        params.learn_coeff = [False, False]
        ## Set boundary condition types
        params.dirichlet_bool = [
            [True, bool(in_params[1]["Dirichlet"]), False, False],
            [True, bool(in_params[1]["Dirichlet"]), False, False],
        ]
        params.cauchy_bool = [
            [False, bool(in_params[1]["Cauchy"]), False, False],
            [False, bool(in_params[1]["Cauchy"]), False, False],
        ]
        ## Set the Dirichlet and Neumann boundary values if necessary, otherwise set = 0
        params.dirichlet_val = [
            [params.solubility, 0.0, 0.0, 0.0],
            [params.solubility, 0.0, 0.0, 0.0],
        ]
        ## Set multiplier for the Cauchy boundary condition if necessary
        ## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
        params.cauchy_mult = [params.cauchy_val, params.cauchy_val]
