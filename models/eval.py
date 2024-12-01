import torch
import numpy as np

import matplotlib.pyplot as plt

from typing import Tuple

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
    def __init__(self, version=0) -> None:
        self.version = version
        
        self.trainer = Trainer()
        self.model = self.trainer.model
        self.classes = self.trainer.get_classes()
        
        self.device = self.trainer.device
        
        self.model.load_state_dict(torch.load(f'fox-vision-ver-{self.version}.pth', weights_only=True))
        self.model.to(self.device)
        
        self.test_loss, self.test_acc = self.trainer.evaluate()
        
    def get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the predictions from the pre-trained model. In particular,
        it returns the image data, the labels, and the confidence level of the
        model, which is given as the output from from the softmax function.

        Returns:
            Tuplep[torch.Tensor, torch.Tensor, torch.Tensor]: image data, labels, probabilities
        """
        
        images = []
        labels = []
        probs = []
        
        self.model.eval()
        
        with torch.no_grad():
            for inputs, label in self.trainer.test_dl:
                inputs = inputs.to(self.trainer.device)
                label = label.to(self.trainer.device)
                
                y_pred = self.model(inputs)
                
                y_prob = softmax(y_pred, dim=-1)
                
                images.append(inputs)
                labels.append(label)
                probs.append(y_prob)
        
        images = torch.cat(images)
        labels = torch.cat(labels)
        probs = torch.cat(probs)
      
        return images, labels, probs
    
    def plot_confusion_matrix(self) -> None:
        """ Plots the confusion matrix of the predictions. """
        _, labels, probs = self.get_predictions()
        
        pred_labels = torch.argmax(probs, 1)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        
        #  need to do this because torch can't convert cuda device tensor to numpy
        #  so need to use cpu() to copy tensor to host memory first
        cm = confusion_matrix(labels.cpu(), pred_labels.cpu())
        
        # TODO: come back and fix this hard-coding nonsense
        cm = ConfusionMatrixDisplay(cm, display_labels=['arctic-fox', 'fox-girl', 'red-fox']) 
        cm.plot(values_format='d', cmap='Blues', ax=ax)
        
    def correct_incorrect_examples(self):
        images, labels, probs = self.get_predictions()
        pred_labels = torch.argmax(probs, 1)
        
        corrects = torch.eq(labels, pred_labels)
        
        incorrects = []
        
        for image, label, prob, correct in zip(images, labels, probs, corrects):
            if not correct:
                incorrects.append((image, label, prob))
                
        incorrects.sort(
                reverse=True,
                key=lambda x: torch.max(x[2], dim=0).values
            )
        
        return corrects, incorrects
    
    def plot_most_incorrect(self, num_images) -> None:
        """Plots the images that the model was most confident about, 
        but predicted incorrectly. The confidence was determined using the 
        softmax function

        Args:
            num_images (int): number of images to be displayed 
        """
        rows = int(np.sqrt(num_images))
        cols = int(np.sqrt(num_images))
        
        _, incorrects = self.correct_incorrect_examples()
        
        fig = plt.figure(figsize=(25, 20))
        
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i+1)
            
            image, true_label, probs = incorrects[i]
            
            image = image.permute(1, 2, 0)
            true_prob = probs[true_label]
            incorrect_prob, incorrect_label = torch.max(probs, dim=0)
            
            true_class = self.classes[true_label]
            
            incorrect_class = self.classes[incorrect_label]
            
            #  once again, need to do this because torch can't convert cuda device tensor to numpy
            ax.imshow(image.cpu().numpy().astype(int))
            ax.set_title(
                f'true label: {true_class} ({true_prob:.3f}) \n'
                f'pred label: {incorrect_class} ({incorrect_prob:.3f})'
            )
            ax.axis('off')
        
        fig.subplots_adjust(hspace=0.4)
       
        
if __name__ == '__main__':
    eval = ModelEval()
    
    eval.plot_confusion_matrix()
    
    eval.plot_incorrect(num_images=35)
    
    print(eval.test_loss, eval.test_acc)
    
    plt.show()
    