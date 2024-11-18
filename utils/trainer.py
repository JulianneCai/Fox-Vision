from utils.nn import FoxCNN

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable


class Trainer:
    def __init__(self):
        self.model = FoxCNN()  #  fox convolutional neural network
        self.loss = nn.BCELoss()  #  binary cross-entropy loss
        self.optimiser = Adam(self.model.parameters(), lr=0.001, weight_decay=.0001)
         
    def save_model(self):
        path = './fox.pth'
        torch.save(self.model.state_dict(), path)
        
    def load_data(self):
        raise NotImplementedError
        
    def test_accuracy(self):
        self.model.eval()
        
        accuracy = 0
        total = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        