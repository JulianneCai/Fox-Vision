import torch

import matplotlib.pyplot as plt

from torch.nn.functional import softmax

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
try:
    from models.trainer import Trainer
except ModuleNotFoundError:
    from trainer import Trainer


class ModelEval:
    """Contains methods for evaluating the model performance.
    
    Args:
        model (torch.nn.Module): the CNN model
        version (int): the version of the model
    """
    def __init__(self, version=0):
        self.version = version
        
        self.trainer = Trainer()
        self.model = self.trainer.model
        
        self.model.load_state_dict(torch.load(f'fox-vision-ver-{self.version}.pth'))
        
        self.test_loss, self.test_acc = self.trainer.evaluate()
        
    def get_predictions(self):
        
        images = []
        labels = []
        probs = []
        
        self.model.eval()
        
        with torch.no_grad():
            for inputs, label in self.trainer.test_dl:
                inputs = inputs.to(self.trainer.device)
                label = label.to(self.trainer.device)
                
                y_pred, _ = self.model(inputs)
                
                y_prob = softmax(y_pred, dim=-1)
                
                images.append(inputs)
                labels.append(label)
                probs.append(y_prob)
        
        images = torch.cat(images)
        labels = torch.cat(labels)
        probs = torch.cat(probs)
      
        return images, labels, probs
    
    def plot_confusion_matrix(self):
        _, labels, probs = self.get_predictions()
        
        pred_labels = torch.argmax(probs, 1)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        
        cm = confusion_matrix(labels, pred_labels)
        
        # TODO: come back and fix this hard-coding nonsense
        cm = ConfusionMatrixDisplay(cm, display_labels=['arctic-fox', 'fox-girl', 'red-fox']) 
        cm.plot(values_format='d', cmap='Blues', ax=ax)
        
if __name__ == '__main__':
    
    eval = ModelEval()
    
    eval.plot_confusion_matrix()
    
    print(eval.test_loss, eval.test_acc)
    
    plt.show()
    