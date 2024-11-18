import torch.nn as nn
import torch.nn.functional as F


class FoxCNN(nn.Module):
    def __init__(self):
        super(FoxCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(num_features=12)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(num_features=24)
        
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(num_features=24)
        
        self.fc1 = nn.Sigmoid()
        
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)
        
        return output