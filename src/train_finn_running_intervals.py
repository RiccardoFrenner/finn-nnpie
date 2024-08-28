"""Train mean FINN models for different time intervals."""

import argparse
import subprocess
from pathlib import Path

import numpy as np


def main(ret_type: str, max_epochs: int, step_size: int):
    data_in_dir = Path("data/FINN_forward_solver/")

    # Directory containing the y_train_path files
    input_dir = data_in_dir / f"retardation_{ret_type}/sub_intervals_{step_size}"

    # Dataset generation
    input_dir.mkdir(exist_ok=True, parents=True)
    c = np.load(data_in_dir / f"retardation_{ret_type}/c_train.npy")
    for i in range(10**9):
        start = i * step_size
        end = start + step_size
        arr = c[start:end]
        if len(arr) != step_size:
            break
        c_sub_path = input_dir / f"c_{i}.npy"
        if c_sub_path.exists():
            continue
        np.save(input_dir / f"c_{i}.npy", arr)

    # Directory for output
    output_base_dir = Path(
        f"data_out/{ret_type}/finn_running_intervals_stepsize_{step_size}_epochs_{max_epochs}"
    )
    output_base_dir.mkdir(exist_ok=True)

    # Gather all y_train_path files in the input directory
    y_train_paths = [input_dir / f"c_{i}.npy" for i in range(12)]

    # Create a list of commands to be executed in parallel
    commands = []
    for y_train_path in y_train_paths:
        output_dir = output_base_dir / y_train_path.stem
        command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx {step_size} --seed 34956765 --max_epochs {max_epochs}"
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
    parser.add_argument("--step_size", type=int, default=30)
    args = vars(parser.parse_args())
    main(**args)
