import argparse
import json
import pprint
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import params
from common import ConcentrationPredictor, create_mlp

cfg = params.Parameters()

def main(
    y_train_path: Path,
    output_dir: Path,
    train_split_idx: int | None = None,
    max_epochs: int = 100,
    seed: int | None = None,
    c_field_seed: int | None = None,
    dropout: int = 0,
    skip: int = 0,
    lr=0.1,
):
    if seed is None:
        seed = int(time.time()) % 10**8

    input_dir = {
        "y_train_path": str(y_train_path),
        "output_dir": str(output_dir),
        "train_split_idx": train_split_idx,
        "max_epochs": max_epochs,
        "seed": seed,
        "c_field_seed": c_field_seed,
        "dropout": dropout,
        "skip": skip,
    }
    pprint.pprint(input_dir)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    t = torch.linspace(0.0, cfg.T, cfg.Nt).float()
    if train_split_idx is None:
        train_split_idx = cfg.Nt
    t_train = t[:train_split_idx-skip].clone()  # we always have to start at t=0 for odeint
    print(f"{t_train.shape=}")

    Y = torch.from_numpy(np.load(y_train_path)).float().unsqueeze(-1)
    Y_train = Y[skip:train_split_idx].clone()

    num_vars = 2
    assert Y.shape == (
        t.shape[0],
        num_vars,
        cfg.Nx,
        1,
    ), f"{Y.shape} != {(t.shape[0], num_vars, cfg.Nx, 1)}"
    assert Y_train.shape[0] == train_split_idx - skip, f"{Y_train.shape[0]} != {train_split_idx-skip}"

    u0 = Y_train[0].clone()
    model = ConcentrationPredictor(
        u0=u0,
        cfg=cfg,
        ret_inv_funs=[
            create_mlp([1, 15, 15, 15, 1], nn.Tanh(), nn.Sigmoid(), dropout=dropout),
            None,
        ],
    )

    clear_dirs = False
    if clear_dirs and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists() and len([p for p in output_dir.glob("*") if p.is_dir()]) > 0:
        raise ValueError(
            f"Folder {output_dir} already exists and is not empty: {list(output_dir.glob('*'))}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "input.json", "w") as f:
        json.dump(input_dir, f, indent=4)

    np.save(output_dir / "c_full.npy", Y.numpy())

    # Train the model
    model.run_training(
        t=t_train,
        u_train=Y_train,
        out_dir=output_dir,
        max_epochs=max_epochs,
        c_field_seed=c_field_seed,
        lr=lr,
    )

    model.eval()
    with torch.no_grad():
        c_predictions = model(t)
    np.save(
        output_dir / "c_full_predictions.npy", c_predictions.detach().numpy()
    )

    return model, t_train, Y_train


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
    parser.add_argument(
        "--skip",
        type=int,
        help="Number of time steps to skip in the training data.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate.",
    )
    args = vars(parser.parse_args())
    model, t_train, Y_train = main(**args)

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
    np.save(
        args["output_dir"] / "c_train_predictions.npy", c_predictions.detach().numpy()
    )
