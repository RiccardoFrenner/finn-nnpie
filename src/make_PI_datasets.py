from pathlib import Path

import numpy as np

MODE = "pos"

base_dir = Path("data_out").resolve()
res_net_in_path = base_dir / "residual_training_data"
res_net_out_path = base_dir / "residual_nets_output"


X_train_diss = np.load(res_net_in_path / f"X_{MODE}_train_diss.npy")
X_train_tot = np.load(res_net_in_path / f"X_{MODE}_train_tot.npy")
X_test_diss = np.load(res_net_in_path / f"X_{MODE}_test_diss.npy")
X_test_tot = np.load(res_net_in_path / f"X_{MODE}_test_tot.npy")

true_residuals_diss = np.load(res_net_in_path / f"Y_{MODE}_train_diss.npy").reshape(
    -1, 1
)
true_residuals_tot = np.load(res_net_in_path / f"Y_{MODE}_train_tot.npy").reshape(-1, 1)

pred_residuals_diss = np.load(res_net_out_path / f"Y_{MODE}_test_diss.npy")
pred_residuals_tot = np.load(res_net_out_path / f"Y_{MODE}_test_tot.npy")


# Create finn dataset by adding the predicted and true residuals on the predicted (mean) finn dataset
X_finn_diss = np.concatenate([X_train_diss, X_test_diss], axis=0)
sort_indices = np.argsort(X_finn_diss[:, 0], kind="stable")  # sort time dimension

full_residuals_diss = np.concatenate(
    [true_residuals_diss, pred_residuals_diss], axis=0
)[sort_indices]
full_residuals_tot = np.concatenate([true_residuals_tot, pred_residuals_tot], axis=0)[
    sort_indices
]

import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
fig.suptitle("Residuals")
ax1.pcolor(full_residuals_diss.reshape(51, 26))
ax2.pcolor(full_residuals_tot.reshape(51, 26))

Y_finn_mean_diss = np.load(base_dir / "c_predictions.npy")[:, 0, ...].reshape((-1, 1))
Y_finn_mean_tot = np.load(base_dir / "c_predictions.npy")[:, 1, ...].reshape((-1, 1))
residual_medians = np.load(base_dir / "residual_medians.npy")
print(f"{Y_finn_mean_diss.shape=}")
print(f"{full_residuals_diss.shape=}")
print(f"{(full_residuals_diss + Y_finn_mean_diss).shape=}")
Y_finn_diss = full_residuals_diss + Y_finn_mean_diss + residual_medians[0]
Y_finn_tot = full_residuals_tot + Y_finn_mean_tot + residual_medians[1]


X_finn = X_finn_diss[sort_indices].copy()
Y_finn = np.concatenate([Y_finn_diss, Y_finn_tot], axis=1)[sort_indices].copy()

out_dir = base_dir / "finn_PI_datasets"
out_dir.mkdir(exist_ok=True, parents=True)
np.save(out_dir / f"X_{MODE}_finn.npy", X_finn)
np.save(out_dir / f"Y_{MODE}_finn.npy", Y_finn)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
print(Y_finn.shape)
fig.suptitle("Y_FINN")
ax1.pcolor(Y_finn[:, 0].reshape((51, 26)))
ax2.pcolor(Y_finn[:, 1].reshape((51, 26)))


# TODO: Sorting is not needed, right?
# Y_finn = np.concatenate([Y_finn_diss, Y_finn_tot], axis=0)[diss_sort_indices]
# X_finn = X_finn_diss[diss_sort_indices]


# X_finn_tot = np.concatenate([X_train_tot, X_test_tot], axis=0)
# diss_sort_indices = np.argsort(X_finn_diss, axis=0)
# tot_sort_indices = np.argsort(X_finn_tot, axis=0)

# np.testing.assert_array_equal(
#     X_finn_diss[diss_sort_indices], X_finn_tot[diss_sort_indices]
# )
# np.testing.assert_array_equal(
#     X_finn_diss[tot_sort_indices], X_finn_tot[tot_sort_indices]
# )

plt.show()
