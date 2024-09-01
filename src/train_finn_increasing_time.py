"""Train mean FINN models for different time intervals."""

import argparse
import subprocess
from pathlib import Path

from common import random_fixed_length_seed

def main(ret_type: str, max_epochs: int):
    # Directory containing the y_train_path files
    y_train_path = Path(f"data/FINN_forward_solver/retardation_{ret_type}/c_train.npy")
    # Directory for output
    output_base_dir = Path(f"data_out/{ret_type}/finn_increasing_time")
    output_base_dir.mkdir(exist_ok=True, parents=True)

    # Create a list of commands to be executed in parallel
    commands = []
    for i in range(1, 30):
        n_steps = 10*i
        output_dir = output_base_dir / f"{random_fixed_length_seed()}_finn_increasing_time"
        command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx {n_steps} --seed 34956765 --max_epochs {max_epochs}"
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
    args = vars(parser.parse_args())
    main(**args)
