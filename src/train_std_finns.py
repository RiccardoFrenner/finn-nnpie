"""Train +/- FINN models for multiple quantiles with same seed."""

# TODO: Rename this file to: train_finn_stds
import subprocess
import argparse
from pathlib import Path

from common import random_fixed_length_seed

def main(ret_type: str, max_epochs: int, train_split_idx: int):
    # Directory containing the y_train_path files
    input_dir = Path(f"data_out/{ret_type}/default_finn/finn_stds_input")
    # Directory for output
    output_base_dir = Path(f"data_out/{ret_type}/default_finn/finn_stds_output")
    output_base_dir.mkdir(exist_ok=True)

    # Gather all y_train_path files in the input directory
    y_train_paths = list(input_dir.glob("*"))

    # Create a list of commands to be executed in parallel
    commands = []
    for y_train_path in y_train_paths:
        output_dir = output_base_dir / f"{random_fixed_length_seed()}_{y_train_path.stem}_finn_stds"
        command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx {train_split_idx} --seed 34956765 --max_epochs {max_epochs}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ret_type", type=str, default="langmuir")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--train_split_idx", type=int, default=51)
    args = vars(parser.parse_args())
    main(**args)