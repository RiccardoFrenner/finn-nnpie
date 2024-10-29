"""Train 3 neural networks to obtain a prediction interval for the data."""

import dataclasses as dc
import time
from pathlib import Path
from typing import Literal, Optional, TypeVar, Callable

import keras
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

import utils
import utils.commandline
from utils.files import MySmartFile, MySmartFolder, MyDir, load_json, save_json

BLOCK = True
T = TypeVar("T")


@dc.dataclass
class TrainingParams:
    max_epochs: int = 100
    stop_early: bool = True
    initial_learning_rate_mean: float = 1e-2
    initial_learning_rate_std: float = 1e-2
    batch_size: Optional[int] = 16  # if None, batch size equals dataset size
    loss_mean: str = "mse"
    loss_std: str = "mse"
    optimizer: str = "adam"
    validation_fraction: float = 0.2
    learning_rate_schedule: str = "constant"
    n_neurons_per_layer: list[int] = dc.field(default_factory=lambda: [64, 64])
    activation_mean: str = "tanh"
    activation_std: str = "relu"  # relu works better but is not so smooth
    positivity_method: str = "sqrt_sqr"

    def get_initial_learning_rate(self, model_type: str) -> float:
        if model_type == "mean":
            return self.initial_learning_rate_mean
        if model_type == "std":
            return self.initial_learning_rate_std
        else:
            raise ValueError(f"Invalid model type '{model_type}'")

    def get_loss(self, model_type: str) -> str:
        if model_type == "mean":
            return self.loss_mean
        if model_type == "std":
            return self.loss_std
        else:
            raise ValueError(f"Invalid model type '{model_type}'")

    def get_activation(self, model_type: str) -> str:
        if model_type == "mean":
            return self.activation_mean
        if model_type == "std":
            return self.activation_std
        else:
            raise ValueError(f"Invalid model type '{model_type}'")


@dc.dataclass
class ExperimentParams:
    # TODO: Check that each param is used
    out_dir: Path
    x_data_path: Path
    y_data_path: Path
    quantiles: list[float] = dc.field(default_factory=lambda: [0.95, 0.9, 0.85, 0.8])
    seed: int = dc.field(default_factory=lambda: time.time_ns() % 10**9)
    debug: bool = False
    visualize: bool = True
    load_checkpoint: bool = True
    verbose: int = 0


def read_commandline():
    parser = utils.commandline.parser_from_dataclasses(
        [ExperimentParams, TrainingParams],
        positional_args={"out_dir", "x_data_path", "y_data_path"},
    )
    clargs = vars(parser.parse_args())
    params = (
        utils.commandline.dict_to_dataclass(TrainingParams, clargs, consume=True),
        utils.commandline.dict_to_dataclass(ExperimentParams, clargs, consume=True),
    )
    assert len(clargs) == 0
    return params


def preprocess_data(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        raise ValueError(f"Too many dimensions: {arr.shape}")

    if arr.ndim == 1:
        return arr[:, np.newaxis]

    assert arr.ndim == 2

    return arr


def make_model(layer_sizes: list[int], activation, positivity_method=None):
    model = keras.Sequential(
        [
            keras.Input(shape=(layer_sizes[0],)),
            *[keras.layers.Dense(n, activation=activation) for n in layer_sizes[1:-1]],
            keras.layers.Dense(layer_sizes[-1]),
        ]
    )

    if positivity_method:
        x = model(model.inputs)
        x = {
            "sqrt_sqr": lambda: keras.ops.sqrt(keras.ops.square(x) + 1e-8),  # type: ignore
            "abs": lambda: keras.ops.abs(x),
            "relu": lambda: keras.layers.ReLU()(x),  # bad option
            "softplus": lambda: keras.layers.Activation("softplus")(x),
            "exp": lambda: keras.layers.Activation("exponential")(x),
        }.get(
            positivity_method,
            lambda: ValueError(f"Unknown positivity method: {positivity_method}"),
        )()
        model = keras.Model(model.input, x)

    return model


@dc.dataclass
class TrainingResult:
    model: keras.Model
    x_train: np.ndarray
    y_train: np.ndarray
    x_eval: np.ndarray
    y_eval_pred: np.ndarray
    y_train_pred: np.ndarray

    @property
    def can_plot(self):
        return self.x_train.shape[1] == 1 and self.y_train.shape[1] == 1

    def plot(self, title: str):
        if not self.can_plot:
            return
        plt.figure()
        plt.title(title)
        plt.scatter(self.x_train, self.y_train, alpha=0.1)
        plt.plot(self.x_eval, self.y_eval_pred, "k-", lw=3)
        plt.show(block=BLOCK)


def train_model(
    model,
    x_train,
    y_train,
    x_eval,
    training_params,
    experiment_params,
    model_type: Literal["mean", "std"],
) -> TrainingResult:
    if isinstance(model, keras.Model):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=20,
            min_lr=training_params.initial_learning_rate / 100,
        )

        if experiment_params.debug:
            print("Warning: Running eagerly is slow!")
        model.compile(
            loss=training_params.loss,
            optimizer=keras.optimizers.get(
                {
                    "class_name": training_params.optimizer,  # type: ignore
                    "config": {
                        "learning_rate": training_params.get_initial_learning_rate(
                            model_type
                        )
                    },
                }
            ),
            run_eagerly=experiment_params.debug,
        )

        model.fit(
            x_train,
            y_train,
            epochs=training_params.max_epochs,
            batch_size=training_params.batch_size or x_train.shape[0],
            validation_split=training_params.validation_fraction,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=50,
                    restore_best_weights=True,
                    min_delta=1e-3,  # type: ignore
                ),
                reduce_lr,
            ]
            if training_params.stop_early
            else None,
            verbose=experiment_params.verbose,
        )

    y_eval_pred = model.predict(x_eval)
    y_train_pred = model.predict(x_train)

    return TrainingResult(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval_pred=y_eval_pred,
        y_train_pred=y_train_pred,
    )


class P3innDir(MyDir):
    def __init__(self, path):
        super().__init__(path)

        self.config = MySmartFile(path / "p3inn_params.json")
        self.x_data = MySmartFile(path / "x_data.npy")
        self.y_data = MySmartFile(path / "y_data.npy")
        self.x_eval = MySmartFile(path / "x_eval.npy")
        self.pred_mean = MySmartFile(path / "pred_mean.npy")
        self.pred_median = MySmartFile(path / "pred_median.npy")
        self.loss = MySmartFile(path / "p3inn_loss.txt")

        # self.pred_PIs_dir = MySmartFolder(path / "pred_PIs", "ts1data_{}.npy")
    
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


def pi3nn_compute_PI_and_mean(
    out_dir,
    quantiles: list[float],
    x_data_path=None,
    y_data_path=None,
    seed=None,
    visualize=False,
    passed_net_mean=None,
    max_iter=50000,
    load_from_dir: bool = True,
):
    seed = int(hash(out_dir)) % 2**31 if seed is None else int(seed)

    p3inn_dir = P3innDir(Path(out_dir))

    if x_data_path is None:
        x_data_path = p3inn_dir.x_data.path
    if y_data_path is None:
        y_data_path = p3inn_dir.y_data.path

    main(
        TrainingParams(
            max_epochs=max_iter,
        ),
        ExperimentParams(
            out_dir=out_dir,
            x_data_path=x_data_path,
            y_data_path=y_data_path,
            quantiles=quantiles,
            seed=seed,
            load_checkpoint=load_from_dir,
            visualize=visualize,
        ),
        passed_net_mean=passed_net_mean,
    )

def to_keras_model_proxy(fun):
    class Proxy:
        def __call__(self, x):
            return fun(x)
        def predict(self, x):
            return fun(x)
    return Proxy

def main(
        training_params: TrainingParams, experiment_params: ExperimentParams,
        passed_net_mean=None
        ) -> None:
    print(training_params, experiment_params)

    rng = np.random.default_rng(experiment_params.seed)
    keras.utils.set_random_seed(experiment_params.seed)

    p3inn_dir = P3innDir(experiment_params.out_dir)

    try:
        # load from checkpoint
        if experiment_params.load_checkpoint:
            x_data = preprocess_data(p3inn_dir.x_data.load())
            y_data = preprocess_data(p3inn_dir.y_data.load())
        else:
            raise FileNotFoundError()
    except FileNotFoundError:
        # load, shuffle and save
        x_data = preprocess_data(np.load(experiment_params.x_data_path))
        y_data = preprocess_data(np.load(experiment_params.y_data_path))
        shuffle_indices = np.arange(x_data.shape[0])
        rng.shuffle(shuffle_indices)
        x_data = x_data[shuffle_indices]
        y_data = y_data[shuffle_indices]
        p3inn_dir.x_data.save(x_data)
        p3inn_dir.y_data.save(y_data)

    layer_sizes = (
        [x_data.shape[1]] + training_params.n_neurons_per_layer + [y_data.shape[1]]
    )

    if x_data.shape[1] == 1:
        x_eval = np.linspace(x_data.min(), x_data.max(), 1000).reshape(-1, 1)
    else:
        x_eval = x_data.copy()  # TODO: Should be a grid
    p3inn_dir.x_eval.save(x_eval)

    if passed_net_mean is None:
        if experiment_params.load_checkpoint and p3inn_dir.pred_mean.path.exists():
            # load checkpoint
            def mean_model(x):
                return np.interp(x.squeeze(), x_eval.squeeze(), p3inn_dir.pred_mean.load().squeeze())
            mean_model = to_keras_model_proxy(mean_model)  # type: ignore
        else:
            mean_model = make_model(layer_sizes, activation=training_params.activation_mean)
            mean_model.summary()  # type: ignore
    else:
        mean_model = to_keras_model_proxy(passed_net_mean)  # type: ignore

    mean_result = train_model(
        mean_model, x_data, y_data, x_eval, training_params, experiment_params, "mean"
    )

    mean_result.plot("Mean Model")

    # Compute and split residuals
    mean_residuals = y_data - mean_result.y_train_pred
    median_shift = np.median(mean_residuals)
    y_eval_pred_median = mean_result.y_eval_pred + median_shift
    p3inn_dir.pred_median.save(y_eval_pred_median)

    median_residuals = mean_residuals - median_shift
    print(f"{median_shift=}")
    pos_mask = median_residuals > 0
    neg_mask = median_residuals < 0

    assert (
        abs(np.count_nonzero(pos_mask) - np.count_nonzero(neg_mask)) <= 1
    ), "Residuals are not evenly split"

    n_below = np.count_nonzero(
        y_data < np.interp(x_data, mean_result.x_eval.flat, y_eval_pred_median.flat)
    )
    n_above = np.count_nonzero(
        y_data > np.interp(x_data, mean_result.x_eval.flat, y_eval_pred_median.flat)
    )

    plt.figure()
    plt.scatter(x_data, y_data, alpha=0.1)
    plt.plot(mean_result.x_eval, mean_result.y_eval_pred, "-", label="Mean")
    plt.plot(mean_result.x_eval, y_eval_pred_median, "--", label="Median")
    plt.legend()
    plt.show(block=BLOCK)
    assert (
        abs(n_below - n_above) <= 1
    ), f"Residuals are not evenly split: {abs(n_below - n_above)}"

    std_models = {}
    if experiment_params.load_checkpoint and len(list(p3inn_dir.pred_PIs_dir.iterdir())) > 0:
        median_eval = mean_result.y_eval_pred
        for q, bound_up, bound_down in p3inn_dir.iter_pred_PIs():
            if q < 0.8:
                continue
            bound_up = bound_up.squeeze()
            bound_down = bound_down.squeeze()
            print("Defining std funs from checkpoint", q)

            def net_up(x):
                y = np.interp(x.squeeze(), x_eval.squeeze(), bound_up - median_eval)
                return y.reshape(-1, 1)

            def net_down(x):
                y = np.interp(x.squeeze(), x_eval.squeeze(), median_eval - bound_down)
                return y.reshape(-1, 1)
            
            std_models["up"] = to_keras_model_proxy(net_up)
            std_models["down"] = to_keras_model_proxy(net_down)
            break

    # Train residual models
    residual_results = {}
    for mask, sign, model_type in [(pos_mask, 1, "up"), (neg_mask, -1, "down")]:
        mask = np.squeeze(mask)
        x_data_res = x_data[mask, :]
        y_data_res = sign * median_residuals[mask, :]
        scale_factor = (
            1 / y_data_res.max()
        )  # TODO: scaling is needed but I would like to have it done by the model automatically.
        y_data_res *= scale_factor
        assert x_data_res.ndim == 2
        assert y_data_res.ndim == 2

        if len(std_models) == 0:
            model = make_model(
                layer_sizes,
                activation=training_params.activation_std,
                positivity_method=training_params.positivity_method,
            )
            model.summary()
        else:
            model = std_models[model_type]

        residual_results[model_type] = train_model(
            model, x_data_res, y_data_res, x_eval, training_params, experiment_params, "std"
        )
        residual_results[model_type].plot(f"Residual Model {model_type}")

    quantile = 0.93
    # compute x% interval

    upper_res_fulltrain = residual_results["up"].model(x_data)
    lower_res_fulltrain = residual_results["down"].model(x_data)

    def objective(c):
        upper = mean_result.y_train_pred + c * upper_res_fulltrain
        lower = mean_result.y_train_pred - c * lower_res_fulltrain
        return np.mean((y_data >= lower) & (y_data <= upper)) - quantile

    c = optimize.bisect(objective, 0, 10)

    upper_bound = (
        mean_result.y_eval_pred + c * residual_results["up"].y_eval_pred
    ).squeeze()
    lower_bound = (
        mean_result.y_eval_pred - c * residual_results["down"].y_eval_pred
    ).squeeze()

    x_eval = mean_result.x_eval.squeeze()
    n_inside = np.count_nonzero(
        (y_data < np.interp(x_data, x_eval, upper_bound))
        & (y_data > np.interp(x_data, x_eval, lower_bound))
    )
    print(f"PI contains {n_inside / x_data.shape[0]:.0%}")

    # Plot final result
    plt.figure(figsize=(12, 8))
    plt.title(f"Prediction Interval (c={c:.3f})")
    plt.scatter(x_data, y_data, alpha=0.1, label="Data")
    plt.plot(mean_result.x_eval, mean_result.y_eval_pred, "-", lw=2, label="Mean")
    plt.plot(mean_result.x_eval, y_eval_pred_median, "-", lw=2, label="Median")
    plt.plot(x_eval, upper_bound, "r--", lw=2, label=f"Upper {quantile:.0%}")
    plt.plot(x_eval, lower_bound, "b--", lw=2, label=f"Lower {quantile:.0%}")
    plt.legend()
    plt.show()

    # TODO:
    # - hyperparam optimization


if __name__ == "__main__":
    training_params, experiment_params = read_commandline()
    main(training_params, experiment_params)  # type: ignore
