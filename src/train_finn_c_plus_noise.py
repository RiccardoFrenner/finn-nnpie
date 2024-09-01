"""Train mean FINN models for different noise on c_train."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from common import random_fixed_length_seed


def main(ret_type: str, max_epochs: int, n_timesteps: int):
    # Directory containing the y_train_path files
    input_dir = Path(f"data/FINN_forward_solver/retardation_{ret_type}/c_plus_noise")
    input_dir.mkdir(exist_ok=True, parents=True)

    # generate c data with noise
    rng = np.random.default_rng()
    c_full = np.load(Path(f"data/FINN_forward_solver/retardation_{ret_type}/c_train.npy"))
    sigma_min = 1e-3
    for i in range(1, 9):
        sigma = sigma_min * 2**i
        for j in range(10):
            c_noise = c_full + rng.normal(0, sigma, c_full.shape)
            out_path = input_dir / f"cFullNoise_sigma={sigma}_j={j}.npy"
            np.save(out_path, c_noise)
    
    # Directory for output
    output_base_dir = Path(f"data_out/{ret_type}/finn_c_plus_noise")
    output_base_dir.mkdir(exist_ok=True, parents=True)

    # Gather all y_train_path files in the input directory
    y_train_paths = list(input_dir.glob("cFullNoise*.npy"))

    # Create a list of commands to be executed in parallel
    commands = []
    for y_train_path in y_train_paths:
        output_dir = output_base_dir / f"{random_fixed_length_seed()}_{y_train_path.stem}_finn_c_plus_noise"
        command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx {n_timesteps} --seed 34956765 --max_epochs {max_epochs}"
        commands.append(command)

    # Write the commands to a temporary file
    commands_file = Path("commands.txt")
    with commands_file.open("w") as f:
        for command in commands:
            f.write(f"{command}\n")

    # Execute the commands in parallel using GNU Parallel
    command = f"cat {commands_file} | parallel -j 8 --bar"
    subprocess.run(command, check=True, shell=True)

    # Clean up the temporary file
    commands_file.unlink()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ret_type", type=str, default="langmuir")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--n_timesteps", type=int, default=51)
    args = vars(parser.parse_args())
    main(**args)
