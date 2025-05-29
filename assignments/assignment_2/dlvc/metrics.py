from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        self.confusion_matrix = torch.zeros((self.classes, self.classes), dtype=torch.int64)



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

       ##TODO implement
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("Both Prediction and Target  have to be of Instance torch.Tensor")

        if prediction.ndim != 4:
            raise ValueError("Prediction should have 4 dimensions (b, c, h, w)")
        if target.ndim != 3:
            raise ValueError("Target should have 3 dimensions (b, h, w)")

        b, c, h, w = prediction.shape
        if c != self.classes:
            raise ValueError("Prediction count has to be equal to the number of classes")
        if target.shape[0] != b or target.shape[1] != h or target.shape[2] != w:
            raise ValueError("Target dimensions must match prediction dimensions of batchsize, "
                             "height, and width (b, h, w)")

        pred_labels = torch.argmax(prediction, dim=1)

        for i in range(b):
            pred_i = pred_labels[i].view(-1)
            target_i = target[i].view(-1)

            pred_i, target_i = pred_i[target_i != 255], target_i[target_i != 255]

            index = self.classes * target_i + pred_i
            cm = torch.bincount(index, minlength=self.classes ** 2)
            cm = cm.reshape(self.classes, self.classes)
            self.confusion_matrix += cm
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        ##TODO implement
        return "mIoU: " + str(round(self.mIoU(), 2))
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        ##TODO implement
        if self.confusion_matrix.sum() == 0:
            return 0.0

        true_pos = torch.diag(self.confusion_matrix).float()
        false_pos = self.confusion_matrix.sum(0).float() - true_pos
        false_neg = self.confusion_matrix.sum(1).float() - true_pos
        denominator = true_pos + false_pos + false_neg

        iou = torch.zeros(self.classes, dtype=torch.float32)
        for i in range(self.classes):
            if denominator[i] != 0:
                iou[i] = true_pos[i] / denominator[i]
            else:
                iou[i] = 0.0

        return iou.mean().item()





