import pickle
from typing import Tuple
import os
import numpy as np


from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        FILES = [f"data_batch_{x+1}" for x in range(5)] + ["test_batch"]
        dir_files = os.listdir(self.fdir)
        read_files = []

        if self.fdir is None or not os.path.isdir(self.fdir):
            raise ValueError("[Wrong directory]")
        if sum([1 for file in FILES if file in dir_files]) < 6:
            raise ValueError("[Files in fdir missing]")

        import pickle
        if self.subset == Subset.TRAINING:
            read_files = [f"data_batch_{x+1}" for x in range(4)]
        elif self.subset == Subset.VALIDATION:
            read_files = ["data_batch_5"]
        elif self.subset == Subset.TEST:
            read_files = ["test_batch"]
        else:
            raise ValueError("[Unknown subset value]")

        labels = []
        images = np.empty((0, 32, 32, 3), dtype=np.uint8)
        for file in read_files:
            with open(os.path.join(self.fdir, file), "rb") as f:
                dict = pickle.load(f, encoding="bytes")

                labels += dict[b'labels']
                data  = dict[b'data'].reshape((-1, 3, 32, 32)).transpose(0,2,3,1) .astype(np.uint8)
                images = np.vstack((images, data))
        
        return images, labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        # TODO implement
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        # TODO implement
        return self.images[idx], self.labels[idx]

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        # TODO implement
        return len(self.classes)
