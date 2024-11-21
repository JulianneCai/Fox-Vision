import torch

from torch.optim.lr_scheduler import _LRScheduler



class ExponentialLearningRate(_LRScheduler):
    def __init__(self, optimiser, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLearningRate, self).__init__(optimiser, last_epoch)
        
    def get_lr(self):
        current_iter = self.last_epoch
        r = current_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r 
                for base_lr in self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)    
    
    def __next__(self):
        try: 
            _, (inputs, labels) = next(enumerate(self._iterator), 0)
        except StopIteration:
            self._iterator = iter(self.iterator)
            _, (inputs, labels), *_ = next(enumerate(self._iterator), 0)

        return inputs, labels
    
    def get_batch(self):
        return next(self)
        

class LearningRateFinder:
    def __init__(self, model, optimiser, criterion):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        torch.save(self.model.state_dict(), 'init_params.pt')
        
    def get_model(self):
        return self.model
    
    def get_optimiser(self):
        return self.optimiser
    
    def get_criterion(self):
        return self.criterion
    
    def get_device(self):
        return self.device
       
    def _train_batch(self, iterator: IteratorWrapper):
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
    
    def range_test(self, iterator, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):
        lrs = []
        losses = []
        best_loss = float('inf')
        
        lr_scheduler = ExponentialLearningRate(self.optimiser, end_lr, num_iter)
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
        
        self.model.load_state_dict(torch.load('init_params.pt'))
        
        return lrs, losses
    