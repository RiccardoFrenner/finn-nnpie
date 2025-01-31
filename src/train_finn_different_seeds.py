"""Train mean FINN models for different seeds."""

import argparse
import subprocess
from pathlib import Path

from common import random_fixed_length_seed

def main(ret_type: str, max_epochs: int, n_timesteps: int):
    y_train_path = Path(f"data/FINN_forward_solver/retardation_{ret_type}/c_train.npy")

    output_base_dir = Path(f"data_out/{ret_type}/finn_different_seeds")
    output_base_dir.mkdir(exist_ok=True, parents=True)

    seeds = [random_fixed_length_seed() for _ in range(16)]

    # Create a list of commands to be executed in parallel
    commands = []
    for seed in seeds:
        output_dir = output_base_dir / f"{seed}_finn_different_seeds"
        command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx {n_timesteps} --seed {seed} --max_epochs {max_epochs}"
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