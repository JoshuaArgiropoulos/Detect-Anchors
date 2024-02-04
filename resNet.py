import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_classes, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(ResNet, self).__init__()
        resnet18 = models.resnet18(weights=weights) # Load pre-trained ResNet18 model with specified weights

        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1]) # Use all layers of ResNet18 except the final fully connected layer

        in_features = resnet18.fc.in_features # Get the number of input features for the fully connected layer

        self.fc = nn.Linear(in_features, num_classes)  # Add a fully connected layer with the specified number of output classes
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        x = self.resnet18(x) # Pass input through the ResNet18 layers
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Apply adaptive average pooling to reduce spatial dimensions to (1, 1)
        x = x.view(x.size(0), -1) # Flatten the tensor for the fully connected layer
        x = self.fc(x) # Pass through the fully connected layer
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x
