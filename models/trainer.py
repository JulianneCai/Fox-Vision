import time
import math

import torch
from tqdm import tqdm

import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from torch.optim import Adam

try:
    from utils.processImage import ImageProcessor
except ModuleNotFoundError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from utils.processImage import ImageProcessor



class FoxCNN(nn.Module):
    def __init__(self, output_dim):
        """Convolutional neural network that recognises pictures of foxes

        Args:
            output_dim (int): number of classes 
        """
        super().__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(kernel_size=2),  # kernel_size
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096), # using batch_size 4
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Sigmoid()
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.size(0), -1)
        x = self.classifier(h)
        return x, h
   
        
    
class Trainer:
    def __init__(self):
        self.model = FoxCNN(output_dim=2)  #  convolutional neural network
        self.loss = nn.CrossEntropyLoss()  #  cross-entropy loss
        #  Adam optimiser
        self.optimiser = Adam(self.model.parameters(), lr=0.001, weight_decay=.0001)
        self.batch_size = 64
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.version = 0
        
        self.img_process = ImageProcessor()
        
        self.train_loader = self.img_process.data_loaders['train']
        self.test_loader = self.img_process.data_loaders['test']
        
    def get_version(self):
        return self.version
    
    def set_version(self, version):
        self.version = version
        
    def count_neurons(self):
        """The number of neurons at a given convolutional layer is given by
        floor((spatial_dimension + 2*padding - kernel)/stride + 1).

        Returns:
            int: number of neurons 
        """
        #  dimensions of resized images
        input_size = 64 
        #  3 channels, image of size 64x64, with stride 2
        first_layer = input_size * input_size/2 * input_size/2 
        #  192 output channels, kernel size 3x3, stride 1, padding 1
        #  dimensions halved due to pooling
        second_layer = 192 * input_size/(2**2) * input_size/(2**2) 
        #  384 output channels, kernel size 3x3, stride 1 padding 1
        #  dimension halved again due to pooling
        third_layer = 384 * input_size/(2**3) * input_size/(2**3)
        #  256 output channels, kernel size 3x3, stride 1, padding 1
        fourth_layer = 256 * input_size/(2**4) * input_size/(2**4)
        #  256 output channels, kernel size 3x3, stride 1, padding 1
        fifth_layer = 256 * input_size/(2**5) * input_size/(2**5)
        
        final_layer = 256 * 2 * 2
        
        #  two fully connected layers
        fully_conn_layer1 = fully_con_layer2 = 4096
        
        #  output layer
        output_layer = 2
        
        neurons = first_layer + second_layer + third_layer + fourth_layer + fifth_layer + final_layer \
            + fully_conn_layer1 + fully_con_layer2 + output_layer
            
        return math.floor(neurons)
         
    def count_parameters(self):
        """Returns the number of trainable parameters of the model

        Args:
            model (utils.FoxCNN): the convolutional neural network

        Returns:
            int: the number of trainanble parameters
        """
        return sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        
    def get_model(self):
        """ Returns the model 
        
        Returns:
            utils.FoxCNN: the convolutional neural network 
        """
        return self.model
        
    def update_ver(self):
        """ Updates the version of the model """
        self.version += 1
         
    def save_model(self):
        """ Saves the pre-trained CNN model """
        path = 'fox-brain-ver-' + self.version + '.pth'
        torch.save(self.model.state_dict(), path)
        
    def load_data(self):
        raise NotImplementedError
    
    def calculate_accuracy(self, y_pred, y):
        """Computes the accuracy of the CNN

        Args:
            y_pred (torch.Tensor): predictions generated by the model
            y (torch.Tensor): actual labels of the model

        Returns:
            float: accuracy of the model
        """
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        accuracy = correct.float() / y.shape[0]
        
        return accuracy
        
    def train(self):
        """ Trains the model over one epoch """
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        #  define execution device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #  convert model parameters and buffers to CPU or Cuda
        self.model.to(device)
        for inputs, labels in tqdm(self.train_loader, leave=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            #  zero parameter gradients
            self.optimiser.zero_grad()
            
            #  predict classes using images from the training set
            #  outputs returns a tuple, first entry is the actual output
            outputs, _ = self.model(inputs)
            # _, pred = torch.max(outputs, 1)
            
            #  compute the loss based on model output and real outputs
            loss = self.loss(outputs, labels) 
            
            accuracy = self.calculate_accuracy(outputs, labels)
            
            #  backpropagate the loss
            loss.backward()
            self.optimiser.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)
        
    def evaluate(self):
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        #  eval() turns off the Dropout step in the CNN
        self.model.eval()
        
        with torch.no_grad():
            for (inputs, labels) in tqdm(self.test_loader, leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(inputs)
                
                loss = self.loss(outputs, labels)
                
                accuracy = self.calculate_accuracy(outputs, labels)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                
        return epoch_loss / len(self.test_loader), epoch_accuracy / len(self.test_loader)
    
    def epoch_eval_time(self, start_time, end_time):
        """Displays how long it takes to train each epoch

        Args:
            start_time (time): start time
            end_time (time): end time

        Returns:
            (time, time): minutes, seconds, elapsed
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def train_over_epoch(self, num_epochs):
        """Train model over number of epochs, and 
        saves the model that has the best validation accuracy.

        Args:
            num_epochs (int): number of epochs 
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.monotonic()
        
            train_loss, train_accuracy = self.train()
            val_loss, val_accuracy = self.evaluate()
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
            
            end_time = time.monotonic()
            
            epoch_mins, epoch_secs = self.epoch_eval_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_accuracy * 100:.2f}%')


if __name__ == '__main__':
    trainer = Trainer()
    
    print('{:,}'.format(trainer.count_parameters()))
    print('{:,}'.format(trainer.count_neurons()))
    
    num_epochs = 15
    
    trainer.train_over_epoch(num_epochs=num_epochs)