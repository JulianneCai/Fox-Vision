import os
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import torchvision.transforms as trans


TRAIN_DIR = 'fox-data/train'
VAL_DIR = 'fox-data/val'

class ImageProcessor:
    def __init__(self):
        self.train = []
        self.test = []
        
    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test
    
    def load_images(self):
        for item in os.listdir(TRAIN_DIR):
            image = Image.open(TRAIN_DIR + '/' + item).convert('RGB')
            image = np.array(image)
            self.train.append(image)
        
        for item in os.listdir(VAL_DIR):
            image = Image.open(VAL_DIR + '/' + item).convert('RGB')
            image = np.array(image)
            self.test.append(image)

       
class FoxDataset(Dataset):
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        image = self.X[index]
        X = self.transform(image)
        return X
    
    transform = trans.Compose([
        trans.ToPILImage(),
        trans.Resize(size=(64, 64)),
        trans.ToTensor()
    ])
    
    
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break
            
if __name__ == '__main__':
    batch_size = 64
    
    ip = ImageProcessor()
    
    ip.load_images()
    
    imgset = ip.get_train()
    
    transformed_dataset = FoxDataset(X=imgset)
    
    train_dl = DataLoader(transformed_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    
    for images in train_dl:
        print(images.shape)
    
    # show_batch(train_dl)
    # plt.show()
    