import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np
import typing
from pathlib import Path
import os



from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave("test_1.png", np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_data = CIFAR10Dataset(
        fdir="your_path_to_the_dataset", subset=Subset.TRAINING, transform=transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=False, num_workers=2
    )

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))

def visualize_statitics(path: Path):
    """
    Visualize the statistics of the dataset by reading file and plotting results.    
    Keyword arguments:
    path -- path statitics file e.g: val_log_ResNet18.txt [forma {train|val|test}_log_{model_name}.txt]
    Return: None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    
    # subset = getattr(Subse,path.stem.split("_")[0])
    subset = path.stem.split("_")[0]

    with open(path, "r") as f:
        lines = f.readlines()
    
    # Extract the data from the file
    epochs = []
    loss = []
    mAcc = []
    mPCAcc = []

    for line in lines[1:]:
        epoch, l, mA, mPCA = line.strip().split(",")
        epochs.append(int(epoch))
        loss.append(float(l))
        mAcc.append(float(mA))
        mPCAcc.append(float(mPCA))

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label="Loss")
    plt.plot(epochs, mAcc, label="Mean Accuracy")
    plt.plot(epochs, mPCAcc, label="Mean Per Class Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title(f"{subset} Statistics")
    plt.legend()
    plt.show()

    # plt.savefig(f"{subset}_statistics.png")
    
