try:
    from utils.processImage import ImageProcessor, FoxDataset
except ModuleNotFoundError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from utils.processImage import ImageProcessor, FoxDataset

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import DataLoader


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
    
    
class Trainer:
    def __init__(self):
        self.model = FoxCNN()  #  fox convolutional neural network
        self.loss = nn.BCELoss()  #  binary cross-entropy loss
        self.optimiser = Adam(self.model.parameters(), lr=0.001, weight_decay=.0001)
        self.batch_size = 64
        
        self.version = 0
        
        img_process = ImageProcessor()
        img_process.load_images()
        
        train, test = img_process.get_train(), img_process.get_test()
        trans_train, trans_test = FoxDataset(X=train), FoxDataset(X=test)
       
        self.train_loader = DataLoader(
            trans_train, 
            self.batch_size, 
            shuffle=True, 
            num_workers=3, 
            pin_memory=True
            ) 
        
        self.test_loader = DataLoader(
            trans_test, 
            self.batch_size, 
            shuffle=True, 
            num_workers=3, 
            pin_memory=True
            ) 
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
        
    def count_parameters(self, model):
        return sum(param.numel() for param in model.parameters())
        
    def get_model(self):
        return self.model
        
    def update_ver(self):
        self.version += 1
         
    def save_model(self):
        path = './fox-ver-' + self.version + '.pth'
        torch.save(self.model.state_dict(), path)
        
    def load_data(self):
        raise NotImplementedError
        
    def test_accuracy(self):
        """
        Function to test the model with the test dataset and print accuracy
        for the test images.
        """
        self.model.eval()
        
        accuracy = 0
        total = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for images, labels in enumerate(self.test_loader):
            #  run the model on the test set to predict labels
            outputs = self.model(images.to(device))
            
            #  the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device).sum().item())
        
        #  compute accuracy over all test images            
        accuracy = (100 * accuracy / total)
        
        return (accuracy)
    
    def train(self, num_epochs):
        """
        Training function that loops over data iterator and feeds 
        the inputs to the network and optimise. 
        Args:
            num_epochs (int): number of epochs 
        """
        best_accuracy = 0.0
        
        #  define execution device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #  convert model parameters and buffers to CPU or Cuda
        self.model.to(device)
        for epoch in range(num_epochs):
            running_loss = 0.0
            i = 0
            for inputs, labels in enumerate(self.train_loader):
                #  obtain inputs
                # images = Variable(images.to(device))
                # labels = Variable(labels.to(device))
                
                #  zero parameter gradients
                self.optimiser.zero_grad()
                
                #  predict classes using images from the training set
                outputs = self.model(inputs)
                
                #  compute the loss based on model output and real labels
                loss = self.loss(outputs, labels)
                
                #  backpropagate the loss
                loss.backward()
                
                #  print statistics for every 100 images
                running_loss += loss.item()
                if i % 100 == 99:
                    print ('[%d, %5d] loss: %.3f' % 
                           (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                i += 1
            
            # compute and print average accuracy for this epoch when tested over all test images                
            accuracy = self.test_accuracy()
            print(f'For epoch {epoch + 1}, the test accuracy over the whole set is {accuracy}')
            
            #  save the model that has the best accuracy
            if accuracy > best_accuracy:
                self.save_model()
                best_accuracy = accuracy


if __name__ == '__main__':
    trainer = Trainer()
    
    trainer.train(5)
    
    trainer.test_accuracy()
    
    model = trainer.get_model()
    trainer.save_model()
    