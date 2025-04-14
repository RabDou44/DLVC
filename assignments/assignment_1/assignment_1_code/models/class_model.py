import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

        # assuming that last layer is same as in resnet18
        # this is a bit hacky, but it works for now
        self.final_layer = nn.Linear(net.fc.out_features, 10)  

    def forward(self, x):
        x = self.net.forward(x)
        return self.final_layer(x)

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        ## TODO implement
        path = save_dir / "model.pt" if suffix is None else save_dir / f"{self.net.__class__.__name__}{suffix}.pt"
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        ## TODO implement
        self.net.load_state_dict(torch.load(path))
        self.net.eval()