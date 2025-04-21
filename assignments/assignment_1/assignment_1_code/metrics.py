from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        ## TODO implement
        if len(prediction.shape) != 2 or len(target.shape) != 1:
            raise ValueError("[Invalid shape]")
        
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("[Invalid shape]")
        
        if prediction.shape[1] != len(self.classes):
            raise ValueError("[Invalid shape]")
        
        prediction = torch.argmax(prediction, dim=1)

        # class accuracy prdiction
        self.n_total += len(prediction)
        for i in range(len(prediction)):
            if prediction[i] == target[i]:
                self.n_matching += 1
                self.correct_pred[self.classes[target[i]]] += 1
            self.total_pred[self.classes[target[i]]] += 1


    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """

        ## TODO implement
        header  =  f"Accuracy: {self.accuracy() * 100:.2f} %\n" + f"Per class accuracy: {self.per_class_accuracy() * 100:.2f}\n"
        body = [f"Accuracy for class: {label} is {self.correct_pred[label]/self.total_pred[label]:.2f}" for label in self.classes]
        return header + "\n".join(body) + "\n"
    
    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        ## TODO implement
        return self.n_matching / self.n_total if self.n_total != 0 else 0.0

    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        ## TODO implement
        return sum( [(v / self.total_pred[k]) /len(self.classes) for k, v in self.correct_pred.items()]) if self.n_total != 0 else 0.0
