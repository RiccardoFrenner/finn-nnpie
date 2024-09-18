"""Train mean FINN models for different noise on c_train."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from common import random_fixed_length_seed

exp_name = "c_plus_better_noise"

def main(ret_type: str, max_epochs: int, n_timesteps: int):
    # Directory containing the y_train_path files
    input_dir = Path(f"data/FINN_forward_solver/retardation_{ret_type}/{exp_name}")
    input_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng()
    c_full = np.load(Path(f"data/FINN_forward_solver/retardation_{ret_type}/c_train.npy"))
    c_full_abs = np.abs(c_full)
    assert c_full.shape[0] > 1
    assert c_full.shape[1] == 2
    assert c_full.shape[2] == 26

    # generate c data with noise
    # Test 1: same sigma for (T,Var,X) aka. sigma = sigma()
    # Test 2: different sigma for Var aka. sigma = sigma(Var)
    # Test 3: different sigma for T,Var aka. sigma = sigma(T, Var)
    # Test 4: sigma = sigma(T, Var, X)
    sigma_scaling_factors = [0.5, 1.0, 2.0]  # check half sigma and double sigma to see if better results
    y_train_paths = []
    for j in range(1, 4):
        for s in sigma_scaling_factors:
            for tmax_idx in [51, None]:
                # Test 1
                sigma = s * 0.05 * c_full_abs[:tmax_idx].mean()
                assert np.shape(sigma) == ()
                c_noise = c_full + rng.normal(0, sigma, c_full.shape)
                out_path = input_dir / f"cFullNoise_test1_{tmax_idx}_sigma={sigma:.2e}_s={s}_j={j}.npy"
                np.save(out_path, c_noise)
                y_train_paths.append(out_path)

                # Test 2
                sigma1 = s * 0.05 * c_full_abs[:tmax_idx, 0].mean()
                sigma2 = s * 0.05 * c_full_abs[:tmax_idx, 1].mean()
                assert np.shape(sigma1) == ()
                assert np.shape(sigma2) == ()
                c_noise = c_full.copy()
                c_noise[:, 0] += rng.normal(0, sigma1, c_noise[:, 0].shape)
                c_noise[:, 1] += rng.normal(0, sigma2, c_noise[:, 1].shape)
                out_path = input_dir / f"cFullNoise_test2_{tmax_idx}_sigma1={sigma1:.2e}_sigma2={sigma2:.2e}_s={s}_j={j}.npy"
                np.save(out_path, c_noise)
                y_train_paths.append(out_path)


            # Test 3
            c_noise = c_full.copy()
            for i in range(c_full.shape[0]):
                sigma1 = s * 0.05 * c_full_abs[i, 0].mean()
                sigma2 = s * 0.05 * c_full_abs[i, 1].mean()
                assert np.shape(sigma1) == ()
                assert np.shape(sigma2) == ()
                c_noise[i, 0] += rng.normal(0, sigma1, c_noise[i, 0].shape)
                c_noise[i, 1] += rng.normal(0, sigma2, c_noise[i, 1].shape)
            out_path = input_dir / f"cFullNoise_test3_sigma1={sigma1:.2e}_sigma2={sigma2:.2e}_s={s}_j={j}.npy"
            np.save(out_path, c_noise)
            y_train_paths.append(out_path)

            # Test 4
            sigma = s * 0.05 * c_full_abs.copy()
            noise = rng.normal(0, sigma)
            assert np.shape(sigma) == c_full.shape
            assert np.shape(noise) == c_full.shape
            c_noise = c_full + noise
            out_path = input_dir / f"cFullNoise_test4_s={s}_j={j}.npy"
            np.save(out_path, c_noise)
            y_train_paths.append(out_path)
    
    # Directory for output
    output_base_dir = Path(f"data_out/{ret_type}/finn_{exp_name}")
    output_base_dir.mkdir(exist_ok=True, parents=True)

    # Create a list of commands to be executed in parallel
    commands = []
    for y_train_path in y_train_paths:
        output_dir = output_base_dir / f"{random_fixed_length_seed()}_{y_train_path.stem}_finn_{exp_name}"
        command = f"python src/train_finn.py {y_train_path} {output_dir} --train_split_idx {n_timesteps} --seed 34956765 --max_epochs {max_epochs} --lr 0.01"
        commands.append(command)

    # Write the commands to a temporary file
    commands_file = Path("commands.txt")
    with commands_file.open("w") as f:
        for command in commands:
            f.write(f"{command}\n")

    # Execute the commands in parallel using GNU Parallel
    command = f"cat {commands_file} | parallel -j 6 --bar"
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
