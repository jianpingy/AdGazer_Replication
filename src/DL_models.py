import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import numpy as np
from torchvision import transforms

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Define MLP
        self.fc1 = nn.Linear(2 * (512 * 4), 256)
        self.fc2 = nn.Linear(256, 1)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x1, x2, Location):
        N = x1.shape[0]

        x1 = self.transform(x1.to(torch.float).to(self.device))
        x2 = self.transform(x2.to(torch.float).to(self.device))
        Location = Location.to(torch.float).to(self.device)

        # Process both images through the same ResNet
        f1 = self.resnet[:8](x1)
        h = f1.register_hook(self.activations_hook)
        f1 = self.resnet[8:](f1)
        f2 = self.resnet(x2)

        # Flatten the features
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)

        f_ad = f1*Location[:,0].reshape(N,1) + f2*Location[:,1].reshape(N,1)
        f_context = f1*(1-Location[:,0].reshape(N,1)) + f2*(1-Location[:,1].reshape(N,1))

        # Concatenate the features
        combined = torch.cat((f_ad, f_context), dim=1)

        # Pass through MLP
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)

        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        x = self.transform(x.to(torch.float).to(self.device))
        return self.resnet[:8](x)