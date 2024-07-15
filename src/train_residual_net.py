import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 10000
LR = 7e-4

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid(),  # Output in [0,1]
        )

    def forward(self, x):
        return self.layers(x)


def main():
    parser = argparse.ArgumentParser(description="Train a 3-layer MLP for regression")
    parser.add_argument("x_train", type=Path, help="Path to x_train.npy")
    parser.add_argument("y_train", type=Path, help="Path to y_train.npy")
    parser.add_argument("x_test", type=Path, help="Path to x_test.npy")
    parser.add_argument(
        "output", type=Path, help="Path to save the predictions on x_test"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    x_train = torch.from_numpy(np.load(args.x_train)).float()
    y_train = torch.from_numpy(np.load(args.y_train).reshape(-1, 1)).float()
    x_test = torch.from_numpy(np.load(args.x_test)).float()

    # --- Scale data ---
    y_train_min = y_train.min().detach().numpy()
    y_train_max = y_train.max().detach().numpy()
    y_train = (y_train - y_train_min) / (y_train_max - y_train_min)

    # Determine input and output sizes
    input_size = x_train.shape[1]
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1

    # Initialize model, loss, and optimizer
    model = MLP(input_size, output_size)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # Train the model
    for epoch in range(EPOCHS):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % (EPOCHS // 5) == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # Evaluate on test data
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(x_test)

    # --- Unscale data ---
    predictions = predictions * (y_train_max - y_train_min) + y_train_min
    # Save the predictions
    np.save(args.output, predictions.detach().numpy())
    print(f"Test Set Predictions saved to: {args.output}")

    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 26)
    t = np.linspace(0, 10000, 2001)[:51]

    X, T = np.meshgrid(x, t)
    data = torch.from_numpy(np.hstack([T.reshape(-1, 1), X.reshape(-1, 1)])).float()

    # Full field prediction
    with torch.no_grad():  # Disable gradient calculation
        full_predictions = model(data).detach().numpy()
    full_predictions = (
        full_predictions * (y_train_max - y_train_min) + y_train_min
    )  # Unscale
    full_field_output_path = Path(str(args.output).replace("_test_", "_full_"))
    np.save(full_field_output_path, full_predictions)
    print(f"Full Field Predictions saved to: {full_field_output_path}")

    # Plot the predictions
    y_train = (
        y_train.detach().numpy().reshape(-1, 1) * (y_train_max - y_train_min)
        + y_train_min
    )  # Unscale

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

    axs[0, 0].set_title("Full Field Predictions")
    axs[0, 0].scatter(T.flatten(), X.flatten(), c=full_predictions.flatten())

    axs[0, 1].set_title("Full Field Predictions one after the other")
    x_full = np.concatenate([x_train, x_test], axis=0)
    y_full = np.concatenate([y_train, predictions], axis=0)
    axs[0, 1].scatter(*x_full.T, c=y_full.flatten())
    # axs[0,1].scatter(*x_train.T, c=y_train.flatten())
    # axs[0,1].scatter(*x_test.T, c=predictions.flatten())

    axs[0, 2].set_title("Full Field Predictions")
    axs[0, 2].pcolor(full_predictions.reshape(51, 26).T)

    axs[1, 0].set_title("Test Set Predictions")
    axs[1, 0].scatter(*x_test.T, c=predictions.flatten())

    axs[1, 1].set_title("Train Set Targets")
    axs[1, 1].scatter(*x_train.T, c=y_train.flatten())

    fig.savefig(f"residual_net_{args.output.stem}.png")


if __name__ == "__main__":
    main()
