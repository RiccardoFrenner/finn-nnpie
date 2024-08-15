"""Train mean FINN models for different seeds."""

import random
import subprocess
from pathlib import Path

output_base_dir = Path("data_out/finn_different_seeds")
output_base_dir.mkdir(exist_ok=True)
y_train_path = Path("data/synthetic_data/retardation_freundlich/c_train.npy")

seeds = [random.randint(10**4, 10**8) for _ in range(16)]

# Create a list of commands to be executed in parallel
commands = []
for seed in seeds:
    output_dir = output_base_dir / f"{seed}"
    command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx 51 --seed {seed}"
    commands.append(command)

# Write the commands to a temporary file
commands_file = Path("commands.txt")
with commands_file.open("w") as f:
    for command in commands:
        f.write(f"{command}\n")

# Execute the commands in parallel using GNU Parallel
command = f"cat {commands_file} | parallel -j 8 --bar"
subprocess.run(
    command, check=True, shell=True
)

# Clean up the temporary file
commands_file.unlink()
