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
loss_fun_mean = nn.MSELoss
loss_fun_std = nn.MSELoss


def train_network(
    model, optimizer, scheduler, criterion, train_loader, val_loader, max_epochs: int, mode: str
) -> None:
    regularization_factor = 0.0 if mode == "mean" else 4*1e-9
    n_params = sum(1 for _ in model.parameters())
    assert n_params > 0, n_params
    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss_train = criterion(output, target)
            # l1_penalty = sum(torch.mean(torch.square(param)) for param in model.parameters()) / n_params
            # loss_train += l1_penalty * regularization_factor
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

    y_all_cap = np.logical_and(y_U_cap, y_L_cap)  # y_all_cap
    
    # FIXED: This will always be True, lol (should be an and...)
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
        train_mean_net,
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
        self.train_mean_net = train_mean_net

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
            if train_mean_net or network_type != "mean"
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

    def train(self, train_mean=True, train_stds=True):
        if train_mean:
            print("Training MEAN Network")
            train_network(
                model=self.networks["mean"],
                optimizer=self.optimizers["mean"],
                scheduler=self.schedulers["mean"],
                criterion=loss_fun_mean(),
                train_loader=[(self.x_train, self.y_train)],
                val_loader=[(self.x_valid, self.y_valid)],
                max_epochs=self.configs["Max_iter"],
                mode="mean",
            )

        data_train_up, data_train_down = create_PI_training_data(
            self.networks["mean"], X=self.x_train, Y=self.y_train
        )
        data_val_up, data_val_down = create_PI_training_data(
            self.networks["mean"], X=self.x_valid, Y=self.y_valid
        )

        if train_stds:
            print("Training UP Network")
            train_network(
                model=self.networks["up"],
                optimizer=self.optimizers["up"],
                scheduler=self.schedulers["up"],
                criterion=loss_fun_std(),
                train_loader=[data_train_up],
                val_loader=[data_val_up],
                max_epochs=self.configs["Max_iter"],
                mode="up",
            )
            print("Training DOWN Network")
            train_network(
                model=self.networks["down"],
                optimizer=self.optimizers["down"],
                scheduler=self.schedulers["down"],
                criterion=loss_fun_std(),
                train_loader=[data_train_down],
                val_loader=[data_val_down],
                max_epochs=self.configs["Max_iter"],
                mode="down",
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


class DataPostProcessor:
    def __init__(self, x, y, test_size: float, seed):
        self._scalar_x = StandardScaler()
        self._scalar_y = StandardScaler()

        self.x = self._scalar_x.fit_transform(x)
        self.y = self._scalar_y.fit_transform(y)

        # random split
        xTrainValid, self.xTest, yTrainValid, self.yTest = train_test_split(
            self.x, self.y, test_size=test_size, random_state=seed, shuffle=True
        )
        ## Split the validation data
        self.xTrain, self.xValid, self.yTrain, self.yValid = train_test_split(
            xTrainValid,
            yTrainValid,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

        ### To tensors
        self.x = torch.FloatTensor(self.x)
        self.xTrain = torch.FloatTensor(self.xTrain)
        self.xValid = torch.FloatTensor(self.xValid)
        self.xTest = torch.FloatTensor(self.xTest)
        self.y = torch.FloatTensor(self.y)
        self.yTrain = torch.FloatTensor(self.yTrain)
        self.yValid = torch.FloatTensor(self.yValid)
        self.yTest = torch.FloatTensor(self.yTest)

    def get_postprocessed(self):
        return (
            self.xTrain,
            self.yTrain,
            self.xTest,
            self.yTest,
            self.xValid,
            self.yValid,
        )

    def transform(self, x, y):
        return self._scalar_x.transform(x), self._scalar_y.transform(y)

    def inverse_transform(self, x, y):
        return self._scalar_x.inverse_transform(x), self._scalar_y.inverse_transform(y)


def pi3nn_compute_PI_and_mean(
    out_dir,
    quantiles: list[float],
    x_data_path=None,
    y_data_path=None,
    seed=None,
    visualize=False,
    passed_net_mean=None,
    max_iter=50000,
    load_from_dir: bool=True,
):
    p3inn_dir = P3innDir(Path(out_dir))
    if p3inn_dir.is_done and not load_from_dir:
        return

    train_mean_net = True if passed_net_mean is None else False

    if x_data_path is None:
        x_data_path = p3inn_dir.x_data_path
    if y_data_path is None:
        y_data_path = p3inn_dir.y_data_path

    X_ = np.load(x_data_path)
    Y_ = np.load(y_data_path)
    np.save(p3inn_dir.x_data_path, X_)
    np.save(p3inn_dir.y_data_path, Y_)

    plt.figure()
    plt.plot(X_, Y_, ".")
    plt.savefig(p3inn_dir.image_dir / "input_data.png")
    if not visualize:
        plt.close()

    SEED = int(hash(out_dir)) % 2**31 if seed is None else int(seed)

    data_processor = DataPostProcessor(X_, Y_, test_size=0.1, seed=SEED)
    xTrain, yTrain, xTest, yTest, xValid, yValid = data_processor.get_postprocessed()

    x_eval = np.linspace(data_processor.x.min(), data_processor.x.max(), 100).reshape(
        -1, 1
    )

    net_mean=None
    net_up=None
    net_down=None

    plt.figure()
    if passed_net_mean is not None:
        # the passed mean function does not know about any scaling. So it wants unscaled x values.
        # However, below I only want to deal with scaled values which is why this code here exists.
        def net_mean(x):
            y = passed_net_mean(data_processor._scalar_x.inverse_transform(x))
            y_trafo = data_processor._scalar_y.transform(y)
            return torch.from_numpy(y_trafo).to(torch.float32)

        x_tmp = np.linspace(xTrain.min(), xTrain.max()).reshape(-1, 1)
        plt.plot(x_tmp, net_mean(x_tmp), "k-", label="Given Mean", zorder=100, lw=3)
    elif load_from_dir:
        if p3inn_dir.pred_mean_path.exists():
            train_mean_net = False
            def net_mean(x):
                y = np.interp(data_processor._scalar_x.inverse_transform(x), data_processor._scalar_x.inverse_transform(x_eval), np.load(p3inn_dir.pred_mean_path))
                y_trafo = data_processor._scalar_y.transform(y)
                return torch.from_numpy(y_trafo).to(torch.float32)

            x_tmp = np.linspace(xTrain.min(), xTrain.max()).reshape(-1, 1)
            plt.plot(x_tmp, net_mean(x_tmp), "k-", label="Loaded Mean", zorder=100, lw=3)
    plt.plot(xTrain, yTrain, ".", alpha=0.5)
    plt.plot(xTest, yTest, "x", alpha=0.5)
    plt.plot(xValid, yValid, "d", alpha=0.5)
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
        "Max_iter": max_iter,
    }
    import random

    LR = 1e-2
    DECAY_RATE = 0.96  # Define decay rate
    DECAY_STEPS = 10000  # Define decay steps
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])

    """ Create network instances"""
    net_mean = (
        UQ_Net_mean(configs, num_inputs, num_outputs) if net_mean is None else net_mean
    )
    train_stds=True
    if load_from_dir and len(list(p3inn_dir.pred_PIs_dir.iterdir())) > 0:
        for q, bound_up, bound_down in p3inn_dir.iter_pred_PIs():
            def net_up(x):
                y = np.interp(data_processor._scalar_x.inverse_transform(x), data_processor._scalar_x.inverse_transform(x_eval), bound_up - data_processor._scalar_y.transform(net_mean(data_processor._scalar_x.inverse_transform(x_eval))))
                y_trafo = data_processor._scalar_y.transform(y)
                return torch.from_numpy(y_trafo).to(torch.float32)
            def net_down(x):
                y = np.interp(data_processor._scalar_x.inverse_transform(x), data_processor._scalar_x.inverse_transform(x_eval), bound_down - data_processor._scalar_y.transform(net_mean(data_processor._scalar_x.inverse_transform(x_eval))))
                y_trafo = data_processor._scalar_y.transform(y)
                return torch.from_numpy(y_trafo).to(torch.float32)
            train_stds = False
            break
            
    net_up = UQ_Net_std(configs, num_inputs, num_outputs, net="up") if net_up is None else net_up
    net_down = UQ_Net_std(configs, num_inputs, num_outputs, net="down") if net_down is None else net_down

    # Initialize trainer and conduct training/optimizations
    trainer = CL_trainer(
        configs,
        net_mean,
        net_up,
        net_down,
        train_mean_net=train_mean_net,
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
    trainer.train(train_mean=train_mean_net, train_stds=train_stds)  # training for 3 networks

    preds_full = trainer.eval_networks(data_processor.x, as_numpy=True)
    preds_full["median"] = shift_to_median(preds_full["mean"], data_processor.y)
    median_shift = compute_shift_to_median(preds_full["mean"], data_processor.y)

    print(f"{median_shift=}")
    print((preds_full["median"] - data_processor.y.numpy() > 0).sum())
    print((preds_full["median"] - data_processor.y.numpy() < 0).sum())

    preds_eval = trainer.eval_networks(
        torch.from_numpy(x_eval).to(torch.float32), as_numpy=True
    )
    assert (preds_eval["up"] > 0).all()
    assert (preds_eval["down"] > 0).all()
    preds_eval["median"] = preds_eval["mean"] + median_shift

    def inv_traf_x(x):
        return data_processor._scalar_x.inverse_transform(x)

    def inv_traf_y(y):
        return data_processor._scalar_y.inverse_transform(y)

    for quantile in quantiles:
        c_up, c_down = compute_boundary_factors(
            y_train=yTrain.numpy(),
            network_preds=trainer.eval_networks(xTrain, as_numpy=True),
            quantile=quantile,
            verbose=1,
        )

        assert c_up > 0 and c_down > 0
        upper_PI = preds_eval["median"] + c_up * preds_eval["up"]
        lower_PI = preds_eval["median"] - c_down * preds_eval["down"]

        # save PIs
        np.save(p3inn_dir.x_eval_path, inv_traf_x(x_eval))
        np.save(p3inn_dir.pred_mean_path, inv_traf_y(preds_eval["mean"]))
        np.save(p3inn_dir.pred_median_path, inv_traf_y(preds_eval["median"]))
        np.save(p3inn_dir.get_pred_bound_path(quantile, "up"), inv_traf_y(upper_PI))
        np.save(p3inn_dir.get_pred_bound_path(quantile, "down"), inv_traf_y(lower_PI))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # data points + mean and median predictions + one PI
    ax1.plot(data_processor.x, data_processor.y, ".", alpha=0.2)
    ax1.plot(x_eval.flat, preds_eval["mean"], "r-", label="Mean Prediction")
    ax1.plot(x_eval.flat, preds_eval["median"], "b--", label="Median Prediction")
    ax1.fill_between(
        x_eval.flat,
        lower_PI.flat,
        upper_PI.flat,
        color="grey",
        alpha=0.3,
        label=f"Prediction Interval {quantile:%}",
    )
    ax1.legend()

    # median + all PIs
    ax2.plot(x_eval.flat, preds_eval["mean"], "r-", label="Mean Prediction")
    ax2.plot(x_eval.flat, preds_eval["median"], "b--", label="Median Prediction")
    for i, (q, bound_up, bound_down) in enumerate(p3inn_dir.iter_pred_PIs()):
        ax2.fill_between(
            x=x_eval.flat,
            y1=data_processor._scalar_y.transform(bound_down).flat,
            y2=data_processor._scalar_y.transform(bound_up).flat,
            color=f"C{0}",
            alpha=0.2,
        )

    fig.savefig(p3inn_dir.image_dir / "learned_data.png")
    if not visualize:
        plt.close(fig)

    # Plot residual training results
    data_train_up, data_train_down = create_PI_training_data(
        trainer.networks["mean"], X=trainer.x_train, Y=trainer.y_train
    )

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
    ax1.set_title("Residuals Up")
    ax1.plot(*data_train_up, ".", alpha=0.3)
    ax1.plot(x_eval, preds_eval["up"], "-", lw=3)
    ax2.set_title("Residuals Down")
    ax2.plot(*data_train_down, ".", alpha=0.3)
    ax2.plot(x_eval, preds_eval["down"], "-", lw=3)
    fig.savefig(p3inn_dir.image_dir / "learned_residuals.png")
    if not visualize:
        plt.close(fig)

    p3inn_dir.done_marker_path.touch()
