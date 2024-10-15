import dataclasses
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import bisect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# tanh results in smoother function
activation_fun_mean = torch.tanh
activation_fun_std = torch.tanh
loss_fun = nn.L1Loss


def train_network(
    model, optimizer, scheduler, criterion, train_loader, val_loader, max_epochs: int
) -> None:
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
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss_valid = criterion(output, target)
                val_loss += loss_valid.item()
        val_loss = val_loss / len(val_loader)

        if epoch % max(1, max_epochs // 10) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>6}, Train Loss: {loss_train:.2e}, Validation Loss: {val_loss:.2e}"
            )

        # Update learning rate scheduler
        scheduler.step()


class CL_dataLoader:
    def __init__(self, original_data_path=None, configs=None):
        if original_data_path:
            self.data_dir = original_data_path
        if configs:
            self.configs = configs

    def load(self):
        X = np.load(self.data_dir / "x.npy").reshape(-1, 1)
        Y = np.load(self.data_dir / "y.npy").reshape(-1, 1)
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
    y_all_cap = np.logical_and(y_U_cap, y_L_cap)  # y_all_cap
    # assert y_all_cap.all(), y_all_cap
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
        X_down.shape,
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


def pi3nn_compute_PI_and_mean(
    out_dir, quantiles: list[float], x_data_path=None, y_data_path=None, seed=None, visualize=True
):
    p3inn_dir = P3innDir(Path(out_dir))
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
    if not visualize:
        plt.close()

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

    plt.figure()
    plt.plot(xTrain, yTrain, ".")
    plt.plot(xTest, yTest, "x")
    plt.plot(xValid, yValid, "d")
    plt.savefig(p3inn_dir.image_dir / "transformed_data_split.png")
    if not visualize:
        plt.close()

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
    if not visualize:
        plt.close(fig)


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
    if not visualize:
        plt.close(fig)

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
    if not visualize:
        plt.close(fig)

    p3inn_dir.done_marker_path.touch()