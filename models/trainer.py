import time
import math

from collections import defaultdict

import torch
from tqdm import tqdm

from typing import Tuple

import torch.nn as nn

from torch.optim import Adam
import torchvision.transforms as transforms

try:
    from learningRate import LearningRateFinder
except ModuleNotFoundError:
    from models.learningRate import LearningRateFinder

try:
    from utils.processImage import ImageProcessor, FoxDataset
    from utils.transforms import Rescale, RandomCrop, ToTensor
    from utils.const import DATA_DIR, BATCH_SIZE, IMG_SIZE
except ModuleNotFoundError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from utils.processImage import ImageProcessor, FoxDataset
    from utils.transforms import Rescale, RandomCrop, ToTensor
    from utils.const import DATA_DIR, BATCH_SIZE, IMG_SIZE
    
    

class FoxCNN(nn.Module):
    """ Convolutional neural network that recognises pictures of foxes. 
    The layers are given by Conv -> Maxpool -> ReLU twice, followed 
    by Conv -> ReLU twice, and then another Conv -> Maxpool -> ReLU.

    Args:
        output_dim (int): number of classes 
    """
    def __init__(self, output_dim) -> None:
        super(FoxCNN, self).__init__()
        
        #  convolution layers that learns the features of the image
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        #  fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            # nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        """ Feed-forward step 
        
        Args:
            x (torch.Tensor): input
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class Trainer:
    """ Class that trains a CNN on dataset """
    def __init__(self) -> None:
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.version = 0
        self.transform = transforms.Compose(
            [
                Rescale(IMG_SIZE),
                RandomCrop(IMG_SIZE),
                ToTensor()
            ]
        )
        
        self.dataset = FoxDataset(
            root_dir=DATA_DIR,
            transform=self.transform
        )
       
        self.img_process = ImageProcessor(
            root_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            transform=self.transform
        )
        
        self.train_dl, self.test_dl = self.img_process.train_test_split_dl(
            self.dataset,
            train_size=0.9,
            test_size=0.1,
            shuffle=True,
            num_workers=0 #  non-zero num_workers results in pickling error with SQLite
        )
        #  convolutional neural network
        #  self.img_process.classes gives a list of the classes (arctic-fox, red-fox)
        #  output_dim is dimension of output layer, which is equal to number of classes
        self.model = FoxCNN(output_dim=len(self.img_process.classes))  
        
        #  loss function we want to optimise 
        self.loss = nn.CrossEntropyLoss()  
       
        #  initialises to None
        #  we will use Adam optimiser, but learning rate will be tuned using LearningRateFinder class
        #  (see learningRate.py)
        self.optimiser = None 
        
    def get_classes(self):
        return self.img_process.classes
        
    def count_neurons(self) -> int:
        """ Counts the number of neurons in the CNN.
        
        The output_channels, padding, stride and kernel size are all hardcoded 
        in the CNN, so the numbers here are hardcoded as well. The calculation
        is done explicitly so that it's clear how the number is being calculated

        Returns:
            int: number of neurons 
        """
        
        input_size = 96
        #  3 channels, image of size 96x96, with stride 2
        first_layer = input_size * input_size/2 * input_size/2 
        #  output channels, kernel size 3x3, stride 1, padding 1
        #  dimensions halved due to pooling
        second_layer =  input_size * 3 * input_size/(2**2) * input_size/(2**2) 
        #  output channels, kernel size 3x3, stride 1 padding 1
        #  dimension halved again due to pooling
        third_layer =  input_size * 6 * input_size/(2**3) * input_size/(2**3) 
        #  output channels, kernel size 3x3, stride 1, padding 1
        fourth_layer =  input_size * 4 * input_size/(2**4) * input_size/(2**4)
        #  output channels, kernel size 3x3, stride 1, padding 1
        fifth_layer = input_size * 4 * input_size/(2**5) * input_size/(2**5)
        #  final layer has output size, and then 2x2 kernel
        final_layer = input_size * 4 * 2 * 2
        
        #  two fully connected layers
        fully_conn_layer1 = fully_con_layer2 = input_size ** 2
        
        #  output layer is number of classes
        output_layer = 3
        
        neurons = first_layer + second_layer + third_layer + fourth_layer + fifth_layer + final_layer \
            + fully_conn_layer1 + fully_con_layer2 + output_layer
            
        return math.floor(neurons)
         
    def count_parameters(self) -> int:
        """Returns the number of trainable parameters of the model

        Args:
            model (utils.FoxCNN): the convolutional neural network

        Returns:
            int: the number of trainable parameters
        """
        return sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        
    def get_version(self) -> int:
        """ Returns the current version of the bot """
        return self.version
        
    def update_version(self) -> None:
        """ Updates model version number by incrementing it by one """
        self.version += 1
        
    def get_model(self) -> FoxCNN:
        """ Returns the model 
        
        Returns:
            utils.FoxCNN: the convolutional neural network 
        """
        return self.model
         
    def save_model(self) -> None:
        """ Saves the pre-trained CNN model """
        path = 'fox-vision-ver-' + str(self.version) + '.pth'
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, version) -> FoxCNN:
        """Loads pre-trained model of a specific version
        
        Args:
            version (int): the version of the specific model

        Returns:
            utils.FoxCNN: the model
        """
        path = 'fox-vision-ver' + str(version) + '.pth'
        model = torch.load(path, weights_only=False)
        return model
        
    def _initialise_parameters(self, param) -> None:
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
    
    def _get_optimal_lr(self, step_flag) -> float:
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
        
        optimiser = Adam(self.model.parameters(), lr=START_LR, weight_decay=0.005)
    
        lr_finder = LearningRateFinder(self.model, optimiser, self.loss)
    
        self.model.to(self.device)
    
        self.loss = self.loss.to(self.device)
        
        #  the end LR is 10, and the number of iterations is 100 by default
        #  see learningRate.py for more details
        lrs, losses = lr_finder.range_test(self.train_dl, 
                                           step_flag=step_flag,
                                           num_iter=100)

        lr_dict = defaultdict(float)
    
        for i in range(len(lrs)):
            lr_dict[lrs[i]] = losses[i]
    
        lr = min(lr_dict, key=lr_dict.get)
        
        return lr
    
    def get_optimiser(self, step_flag='exp') -> torch.optim.Optimizer:
        """ Returns the optimiser that we are using, with optimal learning rate
        
        Args:
            step_flag (str): must be one of ['exp', 'lin']. Whether or not to use linear or exponential LR finder. Defaults to 'exp'. 
        
        Returns:
            torch.optim.Optimizer: adam optimiser with optimised learning-rate
        """
        if self.optimiser is None:
            lr = self._get_optimal_lr(step_flag)
            print(f'Optimal LR: {lr}')
            self.optimiser = Adam(
                self.model.parameters(), 
                lr=lr,
                weight_decay=0.005
                )
            
            return self.optimiser
        else:
            return self.optimiser
    
    def calculate_accuracy(self, y_pred, y) -> float:
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
        
    def train(self) -> Tuple[float, float]:
        """Trains the model once. To be used in the main loop, where 
        we train over a number of epochs.

        Returns:
            Tuple[float, float]: loss, validation
        """
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        #  search for optimal learning rate (LR) using linear LR finder
        optimiser = self.get_optimiser(step_flag='exp')
        #  hard-coding optimal LR from LearningRateFinder so that it runs faster
        # optimiser = Adam(self.model.parameters(), lr=0.00001, weight_decay=0.005)
        
        self.model.apply(self._initialise_parameters)
        
        #  convert model parameters and buffers to CPU or Cuda
        self.model.to(self.device)
        
        for inputs, labels in tqdm(self.train_dl, leave=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            #  zero parameter gradients
            optimiser.zero_grad()
            
            #  predict classes using images from the training set
            #  outputs returns a tuple, first entry is the actual output
            outputs = self.model(inputs)
            # _, pred = torch.max(outputs, 1)
            
            #  compute the loss based on model output and real outputs
            loss = self.loss(outputs, labels) 
            
            accuracy = self.calculate_accuracy(outputs, labels)
            
            #  backpropagate the loss
            loss.backward()
            optimiser.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        
        return epoch_loss / len(self.train_dl), epoch_accuracy / len(self.train_dl)
        
    def evaluate(self) -> Tuple[float, float]:
        """Evaluates the model on the testing dataset

        Returns:
            Tuple[float, float]: (loss in this epoch, accuracy in this epoch)
        """
        # loss in this epoch
        epoch_loss = 0.0
        # accuracy in this epoch
        epoch_accuracy = 0.0
        
        #  eval() turns off the Dropout step in the CNN
        self.model.eval()
        
        with torch.no_grad():
            for (inputs, labels) in tqdm(self.test_dl, leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                #  model outputs tuple of (outputs, label). Only want outputs
                outputs = self.model(inputs)
                
                loss = self.loss(outputs, labels)
                
                accuracy = self.calculate_accuracy(outputs, labels)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                
        return epoch_loss / len(self.test_dl), epoch_accuracy / len(self.test_dl)
    
    def epoch_eval_time(self, start_time, end_time) -> Tuple[float, float]:
        """Displays how long it takes to train each epoch

        Args:
            start_time (time): start time
            end_time (time): end time

        Returns:
            Tuple[float, float]: minutes, seconds, elapsed
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def train_over_epoch(self, num_epochs) -> None:
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
