import matplotlib.pyplot as plt
import numpy as np
import typing 
import os

from pathlib import Path
from argparse import ArgumentParser

def visualize_statistics(path: Path, res_path: Path):

    """
    Visualize the statistics of the dataset by reading file and plotting results.    
    Keyword arguments:
    path -- path statitics file e.g: val_log_ResNet18.txt [forma {train|val|test}_log_{model_name}.txt]
    res_path -- path to save the statistics file
    e.g: ./results/val_log_ResNet18.png
    Return: None
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Extract the data from the lines
    data = []
    for line in lines:
        parts = line.replace(" ", "").replace("\n", "")
        parts = line.split(",")
        epoch = int(parts[0])
        loss = float(parts[1])
        accuracy = float(parts[2])
        avg_accuracy = float(parts[3])
        data.append((epoch, loss, accuracy, avg_accuracy))

    # Convert to numpy array for easier indexing
    data = np.array(data)

    # Plotting
    fig, ax = plt.subplots(1,3, figsize=(14, 4), sharex=True)
    epochs = data[:, 0]
    loss = data[:, 1]
    mAcc = data[:, 2]
    mPCAcc = data[:, 3]

    color = "tab:red"
    ax[0].set_ylabel("Loss", color=color)
    ax[0].plot(epochs, loss, color=color)
    ax[0].tick_params(axis="y", labelcolor=color)
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")

    color = "tab:blue"
    ax[1].set_ylabel("mAcc", color=color)
    ax[1].plot(epochs, mAcc, color=color)
    ax[1].tick_params(axis="y", labelcolor=color)
    ax[1].set_title("mAcc") 
    ax[1].set_xlabel("Epochs")

    color = "tab:green"
    ax[2].set_ylabel("mPCAcc", color=color)
    ax[2].plot(epochs, mPCAcc, color=color)
    ax[2].tick_params(axis="y", labelcolor=color)
    ax[2].set_title("mPCAcc")
    ax[2].set_xlabel("Epochs")

    fig.tight_layout()

    with open(res_path / f"{path.stem}.png", "wb") as f:
        plt.savefig(f, format="png")


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize statistics of the dataset")
    parser.add_argument(
        "--path",
        type=str,
        default="./saved_models/Subset.TRAINING_log_ResNet.csv",
        help="Path to the statistics file",
    )
    parser.add_argument(
        "--res_path",
        type=str,
        default="./results/",
        help="Path to save the statistics file",
    )
    args = parser.parse_args()
    path = Path(args.path)
    res_path = Path(args.res_path)

    if not path.exists() or not path.is_file():
        raise ValueError(f"[Path to statistics file does not exist] {path}")
    if not res_path.exists():
        directory = os.path.dirname(res_path)
        os.makedirs(directory, exist_ok=True)
    
    visualize_statistics(path, res_path)

