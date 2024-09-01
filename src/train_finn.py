import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import params as cfg
from common import ConcentrationPredictor, create_mlp


def main(
    y_train_path: Path,
    output_dir: Path,
    train_split_idx: int | None = None,
    max_epochs: int = 100,
    seed: int | None = None,
    c_field_seed: int | None = None,
    dropout: int = 0,
):
    if seed is None:
        seed = int(time.time()) % 10**8

    print(f"Loading data from {y_train_path}")
    print(f"Saving files to {output_dir}")
    print(f"Train split index: {train_split_idx}")
    print(f"Max epochs: {max_epochs}")
    print(f"Seed: {seed}")
    print(f"C-Loss Seed: {c_field_seed}")
    print(f"Dropout: {dropout}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    t = np.linspace(0.0, cfg.T, cfg.Nt)
    t_train = torch.from_numpy(t[:train_split_idx]).float()
    print(f"{t_train.shape=}")

    Y = torch.from_numpy(np.load(y_train_path)).float().unsqueeze(-1)
    num_vars = 2
    assert Y.shape == (
        len(t_train),
        num_vars,
        cfg.Nx,
        1,
    ), f"{Y.shape} != {(len(t_train), num_vars, cfg.Nx, 1)}"

    cfg.model_path = output_dir.resolve()
    clear_dirs = False
    if clear_dirs and cfg.model_path.exists():
        shutil.rmtree(cfg.model_path)
    elif cfg.model_path.exists():
        raise ValueError(f"Folder {cfg.model_path} already exists.")
    cfg.model_path.mkdir(parents=True, exist_ok=True)

    u0 = Y[0].clone()
    model = ConcentrationPredictor(
        u0=u0,
        cfg=cfg,
        ret_inv_funs=[
            create_mlp([1, 15, 15, 15, 1], nn.Tanh(), nn.Sigmoid(), dropout=dropout),
            None,
        ],
    )

    # Train the model
    model.run_training(
        t=t_train, u_full_train=Y, max_epochs=max_epochs, c_field_seed=c_field_seed
    )
    return model, t_train, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a FINN model.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "y_train_path", type=Path, help="Path to concentration field file."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the folder to save the predictions and intermediate results into.",
    )
    parser.add_argument(
        "-s",
        "--train_split_idx",
        type=int,
        help="Index after which to split the training data.",
    )
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--c_field_seed",
        type=int,
        help="If not None only a random subset of the concentration field will be used in the loss computation.",
    )
    parser.add_argument(
        "--dropout",
        type=int,
        help="Dropout rate (in percent) for the R(c) MLP. 0 to disable.",
    )
    args = vars(parser.parse_args())
    model, t_train, Y = main(**args)

    if "dropout" in args and args["dropout"] > 0:
        u = torch.linspace(0, 1, 100).reshape(-1, 1)
        n_ensemble = 200
        rs_with_dropout = np.zeros((n_ensemble, len(u)))
        model.train()
        with torch.no_grad():
            for i in range(n_ensemble):
                rs_with_dropout[i] = model.retardation(u).numpy().reshape(-1)
        np.save(args["output_dir"] / "dropout_retardations.npy", rs_with_dropout)

    model.eval()
    with torch.no_grad():
        c_predictions = model(t_train)
    np.save(args["output_dir"] / "c_predictions.npy", c_predictions)
