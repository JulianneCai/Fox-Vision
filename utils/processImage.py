import os
import torch
from skimage import io
import glob
import sys
import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import datasets
import torchvision.transforms as trans


DATA_DIR = 'fox-data/train'
IMG_SIZE = 64
    

# class ImageProcessor:
#     """Decomposes an image into a matrix of RGB values at each pixel"""
#     def __init__(self):
#         self.batch_size = 4
#         self.transform = trans.Compose([
#             # trans.ToPILImage(),
#             trans.Resize(size=(IMG_SIZE, IMG_SIZE)),
#             trans.ToTensor(),
#             trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
        
#         self.img_datasets = {
#             x: datasets.ImageFolder(os.path.join(DATA_DIR, x), self.transform)
#             for x in ['train', 'test']
#         }
        
#         self.data_loaders = {x: DataLoader(self.img_datasets[x], batch_size=self.batch_size, 
#                                            shuffle=True, num_workers=0)
#                              for x in ['train', 'test']}
        
#         self.dataset_sizes = {x: len(self.img_datasets[x]) for x in ['train', 'test']}
#         self.class_names = self.img_datasets['train'].classes
        
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Rescale(object):
    """Rescales the iamge to a given size

    Args:
        output_size (tuple or int): desired output dimension. If tuple, output is 
        matched to size. If int, smaller of image edges is matched to output_size
        while maintaining aspect ratio

    Returns:
        np.ndarray: RGB matrix of the image
    """
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, image):
        height, width = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        elif isinstance(self.output_size, tuple):
            if len(self.output_size) == 2:
                new_height, new_width = self.output_size
            else:
                raise ValueError(f'expected tuple of length 2, but got tuple of length {len(self.output_size)}.')
        else:
            raise ValueError(f'output_size must be type int or tuple, but got {type(self.output_size)}.')
        
        image = trans.resize(image, (new_height, new_width))
        
        return image


class RandomCrop(object):
    """Randomly crops the sample image

    Args:
        output_size (tuple or int): desired output size. If int, square crop is performed.
    """
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            if len(output_size) == 2:
                self.output_size = output_size
            else:
                raise ValueError(f'expected tuple of length 2, but got tuple of length {len(output_size)}')
        else:
            raise ValueError(f'output_size must be type int or tuple, but got {type(output_size)}')
        
    def __call__(self, image):
        height, width = image.shape[:2]
        
        new_height, new_width = self.output_size
        
        top = np.random.randint(0, height - new_height + 1)
        left = np.random.randint(0, width - new_width + 1)
        
        image = image[
            top: top + new_height,
            left: left + new_width
        ]
        
        return image
    
class ToTensor(object):
    """Converts ndarrays in sample to tensors

    Args:
        image (np.ndarray): RGB matrix of the image
    """
    def __call__(self, image):
        #  numpy image uses H x W x C
        #  PyTorch uses C x H x W
        image = image.transpose((2, 0, 1))
        return image
        
       
class FoxDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """Fox dataset 

        Args:
            root_dir (str): root directory 
            transform (torchvision.transforms, optional): Optional transformations to be applied. 
            Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.class_map = {'red-fox': 1, 'arctic-fox': 0}
        
        self.data = []
    
        file_list = glob.glob(DATA_DIR + '/*')
        for class_path in file_list:
            if sys.platform == 'win32':
                class_name = class_path.split('\\')[-1]
            #  unix systems index their files differently
            else:
                class_name = class_path.split('/')[-1]
            for img_path in glob.glob(class_path + '/*.jpg'):
                self.data.append([img_path, class_name])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_path, class_name = self.data[idx]
            
            image = np.array(Image.open(img_path)).convert('RGB')
            
            class_id = self.class_map[class_name] 
            
            #  convert into tensor
            img_tensor = torch.from_numpy(image)
            #  pytorch uses different dimensions
            img_tensor = img_tensor.permute(2, 0, 1)
            
            #  class_id needs to be tensor otherwise DataLoader gets upset
            class_id = torch.tensor([class_id])
            
            if self.transform:
                image = self.transform(image)
                
            return img_tensor, class_name

        
class ImageProcessor:
    """Decomposes an image into a matrix of RGB values at each pixel"""
    def __init__(self):
        self.batch_size = 4
        self.transform = trans.Compose([
            # trans.ToPILImage(),
            trans.Resize(size=(IMG_SIZE, IMG_SIZE)),
            trans.ToTensor(),
            trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classes = os.listdir(DATA_DIR)
        
        
        self.img_datasets = {
            x: datasets.ImageFolder(os.path.join(DATA_DIR, x), self.transform)
            for x in ['train', 'test']
        }
        
        self.data_loaders = {x: DataLoader(self.img_datasets[x], batch_size=self.batch_size, 
                                           shuffle=True, num_workers=0)
                             for x in ['train', 'test']}
        
        self.dataset_sizes = {x: len(self.img_datasets[x]) for x in ['train', 'test']}
        self.class_names = self.img_datasets['train'].classes
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        

if __name__ == '__main__': 
    classes = os.listdir(DATA_DIR)
    print(classes)
    data = []
    
    file_list = glob.glob(DATA_DIR + '/*')
    for class_path in file_list:
        if sys.platform == 'win32':
            class_name = class_path.split('\\')[-1]
        #  unix systems index their files differently
        else:
            class_name = class_path.split('/')[-1]
        for img_path in glob.glob(class_path + '/*.jpg'):
            data.append([img_path, class_name])
            
    im_path, _ = data[0]
    
    image = io.imread(im_path)
    print(im_path)
    
    print(image.shape)
    
    image2 = np.array(Image.open(im_path).convert('RGB'))
    
    print(type(image2))
    print(image2.shape)
    