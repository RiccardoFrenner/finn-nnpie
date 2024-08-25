"""Train mean FINN models for different time intervals."""

import argparse
import subprocess
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--step_size", type=int, default=30)
args = parser.parse_args()

step_size = args.step_size
max_epochs = args.max_epochs


# Directory containing the y_train_path files
input_dir = Path("data/synthetic_data/retardation_freundlich/sub_intervals")

# Dataset generation
input_dir.mkdir(exist_ok=True, parents=True)
c = np.load("data/synthetic_data/retardation_freundlich/c_train.npy")
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
output_base_dir = Path(f"data_out/finn_running_intervals_stepsize_{step_size}_epochs_{max_epochs}")
output_base_dir.mkdir(exist_ok=True)

# Gather all y_train_path files in the input directory
y_train_paths = [input_dir / f"c_{i}.npy" for i in range(16)]

# Create a list of commands to be executed in parallel
commands = []
for y_train_path in y_train_paths:
    output_dir = output_base_dir / y_train_path.stem
    command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx 30 --seed 34956765 --max_epochs {max_epochs}"
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
