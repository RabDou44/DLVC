import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        # check if net has 10 classes in the last layer
        self.net = net

    def forward(self, x):
        return self.net.forward(x)

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        name_model = self.net.__class__.__name__

        ## TODO implement
        path = save_dir / f"{name_model}" if suffix is None else save_dir / f"{name_model}{suffix}.pth"
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        ## TODO implement
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()