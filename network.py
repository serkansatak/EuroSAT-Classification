from torch import nn
import torch
import numpy as np
from typing import Union
from config import *
import torch.nn.functional as F
import torchsummary
import torchensemble
import os


class Net(nn.Module):
    
    __default_config__ = {
        "input_size": 64,
        "labels": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'],
        "loss_function": "crossentropy",
        'dropout': 0.2,
        "pretrained_model": None
    }

    
    def __init__(self, config: dict = None):
        super().__init__()
        self.config = self.__default_config__ if config is None else config
        self._device = "cuda" if USE_CUDA else "cpu"
        self.labels = list(self.config["labels"])
        self.num_classes = len(self.labels)
        if self.config["loss_function"] == 'crossentropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown Loss")
            
        # Cheatsheet:
        # Convolutions (Conv2d) need to match the input's number of channels, and can change the number of output channels
        # It's recommended that you stick to 3x3 convolutions with padding=1, which keeps the spatial size of the tensors unchanged
        # Batch normalization (BatchNorm2d) should match the input's number of channels, and do not affect the size
        # BatchNorm2d is used right after Conv2d
        # ReLU activation can be applied to tensors of any size and does not affect its shape
        # Max pooling (MaxPool2d) divides the spatial dimensions of the image. 
        # Stick to MaxPool2d(2, 2) in this exercise, which divides the size of the output by 2 in each axis.

        # Note: Tensors (arrays) here are of shape [B, C, M, N], where:
        # B is the number of samples in a batch
        # C is the number of "color" or feature channels
        # M is the first spatial dimension
        # N is the second spatial dimension

        # Tip: try to keep track of the shape that the tensor will have after each operation. This is necessary for properly 
        # connecting the convolutional layers to the fully connected layers.

        # Input is [B, 3, 64, 64]

        # Input will be reshaped from [B, 8, 16, 16] to [B, 8*16*16] for fully connected layers

        # Note: the final output must have shape [B, n_classes]
        # We're skipping a softmax activation here since we'll be using a loss function that does it automatically
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # [B,32,64,64]
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), # [B,32,32,32]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),# [B,64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # [B,64,16,16]
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=64 * 16 * 16, out_features=128), # [B, 64*16*16]
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=64), # [B, 128]
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=10), # [B, 10]
            #nn.Softmax(-1)
        )

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolution operations
        x = self.convolutions(x)
        # Reshape
        x = x.view(x.shape[0], -1)
        # Apply fully connected operations
        x = self.fully_connected(x)
        return x

    def summary(self):
        if USE_CUDA:
            torchsummary.summary(self.cuda(), (3,64,64), 128)
        else:
            torchsummary.summary(self, (3,64,64), 128)

    def to(self, device, *args, **kwargs):
        return super().to(device, *args, **kwargs)
    
    
    def predict(self, inputs: Union[np.ndarray, torch.Tensor], return_max:bool = False) -> list[str]:
        if isinstance(inputs, np.ndarray):
            data = Net.preprocess_np_input(inputs)
        elif isinstance(inputs, torch.Tensor):
            data = inputs
        else:
            raise ValueError("Input type is neither np.ndarray nor torch.Tensor.")
        data = data.to(self._device)
        preds = torch.softmax(self.forward(data), dim=-1)
        outputs = []
        for scores in preds.detach().cpu().numpy().tolist():
            output = {label: round(score,3) for score,
                      label in zip(scores, self.labels)}
            outputs.append(output)
        
        if return_max:
            outputs = outputs.argmax(1)
        return outputs
        
        
    @staticmethod
    def preprocess_np_input(inputs: np.ndarray) -> torch.Tensor:
        if len(inputs.shape) == 3: ## Assuming [h, w, c]
            tensor_input = torch.from_numpy(
                                np.expand_dims(inputs, axis=0)
                            ).float().permute(0,3,1,2).contiguous()
        elif len(inputs.shape) == 4: ## Assuming shape of [B, h, w, c]
            tensor_input = torch.from_numpy(inputs).float().permute(0,3,1,2).contiguous()
        else:
            raise ValueError("Input dimensions are neither 3 nor 4.")
    
        return tensor_input
    
    @classmethod
    def from_pretrained(cls, model_path: str, *args, **kwargs) -> nn.Module:
        state_dict = torch.load(model_path)
        pretrained_model = cls(config=state_dict['config'], *args, **kwargs)
        pretrained_model.load_state_dict(state_dict["state_dict"])

    def init_weights(self, pretrained_path: str = None):
        if pretrained_path != None:
            self.load_state_dict(pretrained_path)    

    
    def loss(self, y_pred, y_true):
        torch.softmax(y_pred, dim=-1)
        return self.loss_func(y_pred, y_true)



class Ensemble(torchensemble.VotingClassifier):
    def predict_one(self, *x, estimator_num:int, return_dict: bool = False):
        estimator = self.estimators_[estimator_num]
        #outputs = F.softmax(estimator(*x), dim=1) 
        outputs = estimator.predict(*x)
        if not return_dict:
            outputs = np.argmax(np.array(list(outputs.values())), -1)
        return outputs
    
    def predict_with_custom_vote(self, *x, voting_func: function):
        outputs = [
            F.softmax(estimator(*x), dim=1) for estimator in self.estimators_
        ]
        proba = voting_func(outputs)
        return proba
    
    def predict(self, *x, return_max:bool = False):
        outputs = super().predict(*x)
        if return_max:
            output = outputs.argmax(-1)
        else:
            output = outputs
        return output
        
        
        
        
if __name__ == "__main__":
    model = Net()
    model.to(model._device)
    
    ensemble = VotingClassifier(estimator=Net,
                                n_estimators=10)
    
    
    print("done")
    
    input_arr = np.random.rand(64,64,3)
    
    result = model.predict(input_arr)
    print(result)