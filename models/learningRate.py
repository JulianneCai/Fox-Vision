import torch

from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLearningRate(_LRScheduler):
    """ Exponentially increases the learning rate over a number of iterations
    
    Args:
        optimiser (torch.optim.Optimiser): optimiser
        end_lr (float): largest permissible learning rate 
        num_iter (int): number of iterations
        last_epoch (int): index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimiser, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLearningRate, self).__init__(optimiser, last_epoch)
        
    def get_lr(self):
        """ Gets the learning rate

        Returns:
            List<float>: list of all learning rates
        """
        current_iter = self.last_epoch
        r = current_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r 
                for base_lr in self.base_lrs]


class LinearLearningRate(_LRScheduler):
    """Linearly increases the learning rate over a number of iterations
    
    Args:
        optimiser (torch.optim.Optimiser): optimiser
        end_lr (float): largest permissible learning rate 
        num_iter (int): number of iterations
        last_epoch (int): index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimiser, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimiser, last_epoch)
        
    def get_lr(self):
        """Get the learning rate
        
        Returns:
            List<float>: list of all learning rates
        """
        r = self.last_epoch / (self.num_iter - 1)
        
        return [base_lr + r * (self.end_lr - base_lr) 
                for base_lr in self.base_lrs]


class IteratorWrapper:
    """A thing that iterates
    
    Args:
        iterator: thing that iterates
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)    
    
    def __next__(self):
        try: 
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels
    
    def get_batch(self):
        return next(self)
        

class LearningRateFinder:
    """Finds optimal learning rate
    
    Args:
        model (torch.nn.Module): the convolutional neural network
        optimiser (torch.optim.Optimizer): optimiser for loss function
        criterion (torch.nn.something): choice of loss function (e.g. CrossEntropyLoss)
    """
    def __init__(self, model, optimiser, criterion):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        torch.save(self.model.state_dict(), 'init_params.pt')
        
    def get_model(self):
        """ Choice of neural network model """
        return self.model
    
    def get_optimiser(self):
        """ Choice of optimiser """
        return self.optimiser
    
    def get_criterion(self):
        """ Choice of loss function """
        return self.criterion
    
    def get_device(self):
        """ Returns the device that the CNN is using to run """
        return self.device
       
    def _train_batch(self, iterator: IteratorWrapper):
        """ Trains models on batch of data, and then records down the training loss

        Args:
            iterator (IteratorWrapper): the dataloader 

        Returns:
            float: the training loss
        """
        self.model.train()
        self.optimiser.zero_grad()
        
        x, y = iterator.get_batch()
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred, _ = self.model(x) 
       
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        self.optimiser.step()
        
        return loss.item()
    
    def range_test(
        self, 
        iterator, 
        step_flag, 
        end_lr=10, 
        num_iter=100, 
        smooth_f=0.05, 
        diverge_th=5
    ):
        """ Learning rate (LR) range test
        
        The learning rate range test increases the learning rate exponentially, 
        providing information about how well the neural network can be trained over a 
        range of learning rates. 

        Args:
            iterator (IteratorWrapper): the dataloader
            step_flag (str): one of 'exp' or 'lin'. Determines whether to use linear or exponential step size
            end_lr (int, optional): maximum learning rate to test. Defaults to 10.
            num_iter (int, optional): number of iterations. Defaults to 100.
            smooth_f (float, optional): loss smoothing factor. Defaults to 0.05.
            diverge_th (int, optional): stops testing after crossing this threshold. Defaults to 5.

        Returns:
            tuple(List<floats>, List<floats>): tuple of list of learning rates, and a list of training losses
        """
        lrs = []
        losses = []
        best_loss = float('inf')
        lr_scheduler = None
        if step_flag == 'exp':
            lr_scheduler = ExponentialLearningRate(self.optimiser, end_lr, num_iter)
        elif step_flag == 'lin':
            lr_scheduler = LinearLearningRate(self.optimiser, end_lr, num_iter)
        else:
            raise ValueError(f'step_flag parameter must be one of [exp, lin], but got {step_flag}.')
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):
            loss = self._train_batch(iterator)
            lrs.append(lr_scheduler.get_last_lr()[0])
            
            lr_scheduler.step()
            
            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
            
            if loss < best_loss:
                best_loss = loss
            
            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                break
        #  reset model to initial parameters
        self.model.load_state_dict(torch.load('init_params.pt', weights_only=True))
        
        return lrs, losses
    