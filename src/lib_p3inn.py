import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import bisect

activation_fun_mean = torch.tanh
activation_fun_std = torch.tanh
loss_fun = nn.L1Loss


def train_network(
    model, optimizer, scheduler, criterion, train_loader, val_loader, max_epochs: int
) -> None:
    # early_stopper = EarlyStopper(patience=300, verbose=False)

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss_train = criterion(output, target)
            loss_train.backward()
            optimizer.step()

        # Validation phase
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for data, target in val_loader:
        #         output = model(data)
        #         loss_valid = criterion(output, target)
        #         val_loss += loss_valid.item()
        # val_loss = val_loss / len(val_loader)

        if epoch % max(1, max_epochs // 10) == 0 or epoch == 1:
            # print(f"Epoch {epoch}, Validation Loss: {val_loss:.6f}")
            print(f"Epoch {epoch}, Train Loss: {loss_train:.6f}")

        # Update learning rate scheduler
        scheduler.step()

        # Check early stopping condition
        # early_stopper.update(val_loss, model)
        # if early_stopper.early_stop:
        #     print("Early stopping")
        #     break

    # Load the last checkpoint with the best model
    # TODO: This does not actually look better in the plot?
    # model.load_state_dict(torch.load('checkpoint.pt'))


class CL_dataLoader:
    def __init__(self, original_data_path=None, configs=None):
        if original_data_path:
            self.data_dir = original_data_path
        if configs:
            self.configs = configs

    def load(self):
        X = np.load(os.path.join(self.data_dir, "x.npy")).reshape(-1, 1)
        Y = np.load(os.path.join(self.data_dir, "y.npy")).reshape(-1, 1)
        return X, Y

    def getNumInputsOutputs(self, inputsOutputs_np):
        if len(inputsOutputs_np.shape) == 1:
            numInputsOutputs = 1
        if len(inputsOutputs_np.shape) > 1:
            numInputsOutputs = inputsOutputs_np.shape[1]
        return numInputsOutputs


def caps_calculation(network_preds: dict[str, Any], c_up, c_down, Y, verbose=0):
    """Caps calculations for single quantile"""

    if verbose > 0:
        print("--- Start caps calculations for SINGLE quantile ---")
        print("**************** For Training data *****************")

    if len(Y.shape) == 2:
        Y = Y.flatten()

    bound_up = (network_preds["mean"] + c_up * network_preds["up"]).numpy().flatten()
    bound_down = (
        (network_preds["mean"] - c_down * network_preds["down"]).numpy().flatten()
    )

    y_U_cap = bound_up > Y  # y_U_cap
    y_L_cap = bound_down < Y  # y_L_cap

    # FIXME: This will always be True, lol (should be an and...)
    y_all_cap = np.logical_or(y_U_cap, y_L_cap)  # y_all_cap
    assert y_all_cap.all()
    PICP = np.count_nonzero(y_all_cap) / y_L_cap.shape[0]  # 0-1
    MPIW = np.mean(
        (network_preds["mean"] + c_up * network_preds["up"]).numpy().flatten()
        - (network_preds["mean"] - c_down * network_preds["down"]).numpy().flatten()
    )
    if verbose > 0:
        print(f"Num of train in y_U_cap: {np.count_nonzero(y_U_cap)}")
        print(f"Num of train in y_L_cap: {np.count_nonzero(y_L_cap)}")
        print(f"Num of train in y_all_cap: {np.count_nonzero(y_all_cap)}")
        print(f"np.sum results(train): {np.sum(y_all_cap)}")
        print(f"PICP: {PICP}")
        print(f"MPIW: {MPIW}")

    return (
        PICP,
        MPIW,
    )


def optimize_bound(
    *,
    mode: str,
    y_train: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    num_outliers: int,
    c0: float = 0.0,
    c1: float = 1e5,
    maxiter: int = 1000,
    verbose=0,
):
    def count_exceeding_upper_bound(c: float):
        bound = pred_mean + c * pred_std
        f = np.count_nonzero(y_train >= bound) - num_outliers
        return f

    def count_exceeding_lower_bound(c: float):
        bound = pred_mean - c * pred_std
        f = np.count_nonzero(y_train <= bound) - num_outliers
        return f

    objective_function = (
        count_exceeding_upper_bound if mode == "up" else count_exceeding_lower_bound
    )

    if verbose > 0:
        print(f"Initial bounds: [{c0}, {c1}]")

    try:
        optimal_c = bisect(objective_function, c0, c1, maxiter=maxiter)
        if verbose > 0:
            final_count = objective_function(optimal_c)
            print(f"Optimal c: {optimal_c}, Final count: {final_count}")
        return optimal_c
    except ValueError as e:
        if verbose > 0:
            print(f"Bisect method failed: {e}")
        raise e


def compute_boundary_factors(
    *, y_train: np.ndarray, network_preds: dict[str, Any], quantile: float, verbose=0
) -> tuple[float, float]:
    n_train = y_train.shape[0]
    num_outlier = int(n_train * (1 - quantile) / 2)

    if verbose > 0:
        print(
            "--- Start boundary optimizations for SINGLE quantile: {}".format(quantile)
        )
        print(
            "--- Number of outlier based on the defined quantile: {}".format(
                num_outlier
            )
        )

    c_up, c_down = [
        optimize_bound(
            y_train=y_train,
            pred_mean=network_preds["mean"],
            pred_std=network_preds[mode],
            mode=mode,
            num_outliers=num_outlier,
        )
        for mode in ["up", "down"]
    ]

    if verbose > 0:
        print("--- c_up: {}".format(c_up))
        print("--- c_down: {}".format(c_down))

    return c_up, c_down


def create_PI_training_data(
    network_mean, X, Y
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Generate up and down training data by splitting at the median."""
    with torch.no_grad():
        diff_train = Y.reshape(Y.shape[0], -1) - network_mean(X)
        diff_train -= torch.median(diff_train)  # make sure same amount of up and down
        up_idx = diff_train > 0
        down_idx = diff_train < 0

        X_up = X[up_idx.flatten()]
        Y_up = diff_train[up_idx].unsqueeze(1)

        X_down = X[down_idx.flatten()]
        Y_down = -1.0 * diff_train[down_idx].unsqueeze(1)

    # check that number there is at most one more sample in up/down
    # this property should hold since we split at the median
    # if the number of samples is even, the difference will be 0
    assert abs(X_up.shape[0] - X_down.shape[0]) <= 1, (
        abs(X_up.shape[0] - X_down.shape[0]),
        X_up.shape,
    )

    return ((X_up, Y_up), (X_down, Y_down))


class CL_trainer:
    def __init__(
        self,
        configs,
        net_mean,
        net_up,
        net_down,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test=None,
        y_test=None,
        lr=1e-2,
        decay_rate=0.96,
        decay_steps=1000,
    ):
        """Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance"""

        self.configs = configs

        self.networks = {
            "mean": net_mean,
            "up": net_up,
            "down": net_down,
        }
        self.optimizers = {
            network_type: torch.optim.SGD(
                network.parameters(),
                lr=lr,
                # network.parameters(), lr=1e-3, weight_decay=0  # good for mean, bad for stds
                # network.parameters(), lr=1e-3, weight_decay=2e-2  # too strong weight_decay
                # network.parameters(), lr=1e-3, weight_decay=2e-3  # still a bit too strong for mean, but too little for std
            )
            for network_type, network in self.networks.items()
        }

        self.schedulers = {
            network_type: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=decay_steps, gamma=decay_rate
            )
            for network_type, optimizer in self.optimizers.items()
        }

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        print("Training MEAN Network")
        train_network(
            model=self.networks["mean"],
            optimizer=self.optimizers["mean"],
            scheduler=self.schedulers["mean"],
            criterion=loss_fun(),
            train_loader=[(self.x_train, self.y_train)],
            val_loader=[(self.x_valid, self.y_valid)],
            max_epochs=self.configs["Max_iter"],
        )

        data_train_up, data_train_down = create_PI_training_data(
            self.networks["mean"], X=self.x_train, Y=self.y_train
        )
        data_val_up, data_val_down = create_PI_training_data(
            self.networks["mean"], X=self.x_valid, Y=self.y_valid
        )

        print("Training UP Network")
        train_network(
            model=self.networks["up"],
            optimizer=self.optimizers["up"],
            scheduler=self.schedulers["up"],
            criterion=loss_fun(),
            train_loader=[data_train_up],
            val_loader=[data_val_up],
            max_epochs=self.configs["Max_iter"],
        )
        print("Training DOWN Network")
        train_network(
            model=self.networks["down"],
            optimizer=self.optimizers["down"],
            scheduler=self.schedulers["down"],
            criterion=loss_fun(),
            train_loader=[data_train_down],
            val_loader=[data_val_down],
            max_epochs=self.configs["Max_iter"],
        )

    def eval_networks(self, x, as_numpy: bool = False) -> dict[str, Any]:
        with torch.no_grad():
            d = {k: network(x) for k, network in self.networks.items()}
        if as_numpy:
            d = {k: v.numpy() for k, v in d.items()}
        return d


# TODO: Add regularization to optimizer step
# every dense layer had it: regularizers.l1_l2(l1=0.02, l2=0.02)
class UQ_Net_mean(nn.Module):
    def __init__(self, configs, num_inputs, num_outputs):
        super(UQ_Net_mean, self).__init__()
        self.configs = configs
        self.num_nodes_list = self.configs["num_neurons_mean"]

        self.inputLayer = nn.Linear(num_inputs, self.num_nodes_list[0])
        self.fcs = nn.ModuleList()
        for i in range(len(self.num_nodes_list) - 1):
            self.fcs.append(
                nn.Linear(self.num_nodes_list[i], self.num_nodes_list[i + 1])
            )
        self.outputLayer = nn.Linear(self.num_nodes_list[-1], num_outputs)

        # Initialize weights with a mean of 0.1 and stddev of 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.1, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = activation_fun_mean(self.inputLayer(x))
        for i in range(len(self.fcs)):
            x = activation_fun_mean(self.fcs[i](x))
        x = self.outputLayer(x)
        return x


class UQ_Net_std(nn.Module):
    def __init__(self, configs, num_inputs, num_outputs, net=None, bias=None):
        super(UQ_Net_std, self).__init__()
        self.configs = configs
        if net == "up":
            self.num_nodes_list = self.configs["num_neurons_up"]
        elif net == "down":
            self.num_nodes_list = self.configs["num_neurons_down"]

        self.inputLayer = nn.Linear(num_inputs, self.num_nodes_list[0])
        self.fcs = nn.ModuleList()
        for i in range(len(self.num_nodes_list) - 1):
            self.fcs.append(
                nn.Linear(self.num_nodes_list[i], self.num_nodes_list[i + 1])
            )
        self.outputLayer = nn.Linear(self.num_nodes_list[-1], num_outputs)

        # Initialize weights with a mean of 0.1 and stddev of 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.1, std=0.1)
                nn.init.zeros_(m.bias)

        # Custom bias
        if bias is None:
            self.custom_bias = torch.nn.Parameter(torch.tensor([3.0]))
        else:
            self.custom_bias = torch.nn.Parameter(torch.tensor([bias]))

    def forward(self, x):
        x = activation_fun_std(self.inputLayer(x))
        for i in range(len(self.fcs)):
            x = activation_fun_std(self.fcs[i](x))
        x = self.outputLayer(x)
        x = x + self.custom_bias
        x = torch.sqrt(
            torch.square(x) + 1e-8
        )  # TODO: Was 0.2 but not explained why in paper
        return x

def compute_shift_to_median(y, data):
    """Computes the shift v such that y + v has 50% of data below the curve, assuming y and data have same x values."""
    diff = data - y
    return np.median(diff)


def shift_to_median(y, data):
    """Shifts y such that 50% of data lies below the curve, assuming y and data have same x values."""
    return y + compute_shift_to_median(y, data)
