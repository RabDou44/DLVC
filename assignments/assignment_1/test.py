## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os
from pathlib import Path

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = CIFAR10Dataset(args.path, 
                               Subset.TEST,
                               transform=transform)
    
    test_data_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4
    )

    device = ...
    num_test_data = len(test_data)

    model = DeepClassifier(resnet18())
    model.load(args.model_path)
    # model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss
    ### and metrics to terminal after testing is finished
    # ...sss

    for batch_idx, (data, target) in enumerate(test_data_loader):
        # data = data.to(device)
        # target = target.to(device)

        output = model(data)
        loss = loss_fn(output, target)

        test_metric.update(output, target)

        print(f"Batch {batch_idx}/{len(test_data_loader)}: Loss: {loss.item()}, accuracy: {test_metric.accuracy()}, per class accuracy: {test_metric.per_class_accuracy()}")

    with open(args.results_path, "w") as f:
        f.write(f"Test Loss: {loss.item()}\n")
        f.write(f"Test Accuracy: {test_metric.accuracy()}\n")
        f.write(f"Test Per Class Accuracy: {test_metric.per_class_accuracy()}\n")
        f.write(f"Accuracy per class:\n{test_metric.__str__()}\n")


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )
    args.add_argument(
        "-p", "--path", default="./assignments/assignment_1/assignment_1_code/fdir/", type=str, help="path to dataset"
    )
    args.add_argument("-m","--model_path", default="./saved_models//best_model.pth", type=str, help="path to save model")
    args.add_argument("-b","--batch_size", default=128, type=int, help="batch size")
    args.add_argument("-r","--results_path", default="./results/", type=str, help="path to save results")

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    if not Path(args.path).exists():
        raise ValueError("[Path to TEST set does not exist]")
    
    if not Path(args.model_path).exists():
        raise ValueError("[Path to save model does not exist]")

    test(args)
