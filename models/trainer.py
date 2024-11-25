import time
import math

import torch
from tqdm import tqdm

import torch.nn as nn

from torch.optim import Adam

from learningRate import LearningRateFinder

try:
    from utils.processImage import ImageProcessor
except ModuleNotFoundError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from utils.processImage import ImageProcessor



class FoxCNN(nn.Module):
    def __init__(self, output_dim):
        """ Convolutional neural network that recognises pictures of foxes

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
            # nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096), 
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        """ Feed-forward step 
        
        Args:
            x (torch.Tensor): input
        """
        x = self.features(x)
        h = x.view(x.size(0), -1)
        x = self.classifier(h)
        return x, h
    
    
class Trainer:
    """ Class that trains a CNN on dataset """
    def __init__(self):
        self.model = FoxCNN(output_dim=2)  #  convolutional neural network
        self.loss = nn.CrossEntropyLoss()  #  cross-entropy loss
        self.optimiser = None #  set it to None, will optimiser LR later
        
        # self.batch_size = 64
        self.batch_size = 256
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.version = 0
       
        self.img_process = ImageProcessor()
        
        self.train_loader = self.img_process.data_loaders['train']
        self.test_loader = self.img_process.data_loaders['test']
        
    def get_version(self):
        """ Returns the current version of the bot """
        return self.version
        
    def update_version(self):
        """ Updates model version number by incrementing it by one """
        self.version += 1
        
    def get_model(self):
        """ Returns the model 
        
        Returns:
            utils.FoxCNN: the convolutional neural network 
        """
        return self.model
         
    def save_model(self):
        """ Saves the pre-trained CNN model """
        path = 'fox-brain-ver-' + str(self.version) + '.pth'
        torch.save(self.model.state_dict(), path)
        
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
    
    def _initialise_parameters(self, param):
        """Initialise parameters of our model
        
        Initialise parameters by initialising weights from a normal 
        distribution with a standard deviation given by gain/sqrt(fan mode).
        Here, gain = sqrt(2) since we set initlisation function to ReLU. 
        
        Fan mode can be either fan_in, fan_out, which is the number of connections 
        coming into and out of the layer.
        
        For linear layers, we get a normal distribution with standard deviation given by 
        gain * sqrt(2 /(fan_in + fan_out)).
        
        Doing this initialises our input data to have a mean of 0 and a standard deviation of 1

        Args:
            param (torch.nn.parameter.Parameter): parameters of the model 
        """
        if isinstance(param, nn.Conv2d):
            nn.init.kaiming_normal_(param.weight.data, nonlinearity='relu')
            nn.init.constant_(param.bias.data, 0)
        elif isinstance(param, nn.Linear):
            nn.init.xavier_normal_(param.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(param.bias.data, 0)
    
    def _get_optimal_lr(self, step_flag):
        """ Calculates the optimal learning rate (LR) by starting with a small 
        learning rate, in this case 1e-7, and then exponentiall increases it to 
        optimise the loss function. See learningRate.py for more info
        
        At the end, it returns the LR corresponding to the smallest loss
        
        Args:
            step_flag (str): one of 'exp' or 'lin'. Whether to use linear or exponential LR finder.
        
        Returns:
            float: the optimal learning rate
        """
        START_LR = 1e-8
        
        optimiser = Adam(self.model.parameters(), lr=START_LR)
    
        lr_finder = LearningRateFinder(self.model, optimiser, self.loss)
    
        self.model.to(lr_finder.get_device())
    
        self.loss = self.loss.to(lr_finder.get_device())
        
        #  the end LR is 10, and the number of iterations is 100 by default
        #  see learningRate.py for more details
        lrs, losses = lr_finder.range_test(trainer.train_loader, 
                                           step_flag='exp',
                                           num_iter=300)

        lr_dict = {}
    
        for i in range(len(lrs)):
            lr_dict[lrs[i]] = losses[i]
    
        lr = max(lr_dict, key=lr_dict.get)
        
        return lr / 100
    
    def get_optimiser(self, linear=False):
        """ Returns the optimiser that we are using, with optimal learning rate
        
        Args:
            linear (bool): whether or not to use linear LR finder. Defaults to False
        
        Returns:
            torch.optim.Optimizer: adam optimiser with optimised learning-rate
        """
        if self.optimiser is None:
            lr = self._get_optimal_lr(linear)
            print(f'Optimal LR: {lr}')
            self.optimiser = Adam(self.model.parameters(), lr=lr)
            
            return self.optimiser
        else:
            return self.optimiser
    
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
        
        #  search for optimal learning rate (LR) using linear LR finder
        optimiser = self.get_optimiser(linear=False)
        
        #  define execution device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.model.apply(self._initialise_parameters)
        
        #  convert model parameters and buffers to CPU or Cuda
        self.model.to(device)
        
        for inputs, labels in tqdm(self.train_loader, leave=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            #  zero parameter gradients
            optimiser.zero_grad()
            
            #  predict classes using images from the training set
            #  outputs returns a tuple, first entry is the actual output
            outputs, _ = self.model(inputs)
            # _, pred = torch.max(outputs, 1)
            
            #  compute the loss based on model output and real outputs
            loss = self.loss(outputs, labels) 
            
            accuracy = self.calculate_accuracy(outputs, labels)
            
            #  backpropagate the loss
            loss.backward()
            optimiser.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)
        
    def evaluate(self):
        """Evaluates the model on the testing dataset

        Returns:
            tuple(float, float): (loss in this epoch, accuracy in this epoch)
        """
        # loss in this epoch
        epoch_loss = 0.0
        # accuracy in this epoch
        epoch_accuracy = 0.0
        
        #  eval() turns off the Dropout step in the CNN
        self.model.eval()
        
        with torch.no_grad():
            for (inputs, labels) in tqdm(self.test_loader, leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                #  model outputs tuple of (outputs, label). Only want outputs
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
        """Trains the model
        
        Train model over number of epochs, and 
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
            
            print(f'Epoch: {epoch+1:02} | Epoch Eval. Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_accuracy * 100:.2f}%')


if __name__ == '__main__':
    trainer = Trainer()
    
    print(trainer.count_parameters(), trainer.count_neurons())
    
    num_epochs = 15
    
    trainer.train_over_epoch(num_epochs=num_epochs)