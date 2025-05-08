import matplotlib.pyplot as plt
from assignment_1_code.datasets import Subset
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
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(data[:, 0], data[:, 1], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(data[:, 0], data[:, 2], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax3 = ax1.twinx()
    color = "tab:green"
    ax3.set_ylabel("Avg_accuracy", color=color)
    ax3.plot(data[:, 0], data[:, 3], color=color)
    ax3.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title(f"Statistics from {path.name}")
    plt.show()

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
        os.makedirs(res_path, exist_ok=True)
    
    visualize_statistics(path, res_path)

