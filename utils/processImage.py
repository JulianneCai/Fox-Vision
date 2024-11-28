import os
import torch
import glob
import sys
import numpy as np

import skimage

from PIL import Image
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

try:
    from utils.const import DATA_DIR, IMG_SIZE
except ModuleNotFoundError:
    sys.path.append(sys.path[0] + '/..')
    from utils.const import DATA_DIR, IMG_SIZE

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class Rescale(object):
    """Rescales the iamge to a given size

    Args:
        size (tuple or int): desired output dimension. If tuple, output is 
        matched to size. If int, smaller of image edges is matched to size
        while maintaining aspect ratio

    Returns:
        np.ndarray: RGB matrix of the image
    """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        height, width = image.shape[:2]
        
        if isinstance(self.size, int):
            if height > width:
                new_height, new_width = int(self.size * height / width), self.size
            else:
                new_height, new_width = self.size, int(self.size * width / height)
        elif isinstance(self.size, tuple):
            if len(self.size) == 2:
                new_height, new_width = self.size
            else:
                raise ValueError(f'expected tuple of length 2, but got tuple of length {len(self.size)}.')
        else:
            raise ValueError(f'size must be type int or tuple, but got {type(self.size)}.')
        
        image = skimage.transform.resize(image, (new_height, new_width))
        
        return image


class RandomCrop(object):
    """Randomly crops the sample image for numpy.ndarray RGB matrices.
    The PyTorch version of this function only does it for PIL images

    Args:
        size (tuple or int): desired output size. If int, square crop is performed.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            if len(size) == 2:
                self.size = size
            else:
                raise ValueError(f'expected tuple of length 2, but got tuple of length {len(size)}')
        else:
            raise ValueError(f'size must be type int or tuple, but got {type(size)}')
        
    def __call__(self, image):
        """Randomly crops the image

        Args:
            image (numpy.ndarray): the RGB matrix of the image

        Returns:
            numpy.ndarray: the randomly cropped RGB matrix of the image 
        """
        height, width = image.shape[:2]
        
        new_height, new_width = self.size
        
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
        return torch.from_numpy(image)
        
       
class FoxDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """Fox dataset 

        Args:
            root_dir (str): root directory 
            transform (torchvision.transforms, optional): transformations to be applied. 
            Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = os.listdir(root_dir)
        
        encoder = LabelEncoder()
        encoder.fit(self.classes)
        
        #  label-encoded features
        feature_encodings = encoder.transform(self.classes)
        #  name of the feature
        feature_names = encoder.inverse_transform(feature_encodings)
        
        class_map = defaultdict(list)
        
        for i in range(len(self.classes)):
            class_name = str(feature_names[i])
            class_map[class_name] = int(feature_encodings[i])
        
        self.class_map = class_map
        
        self.data = []
    
        file_list = glob.glob(self.root_dir + '/*')
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
        
        #  dtype must be float32 otherwise Conv2d will complain
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        
        class_id = self.class_map[class_name] 
        
        #  class_id needs to be tensor otherwise DataLoader gets upset
        class_id = torch.tensor(class_id)
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_id

        
class ImageProcessor:
    """Decomposes an image into a matrix of RGB values at each pixel"""
    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        
        if isinstance(img_size, int) or isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise ValueError(f'expected img_size to be of type int or tuple(int, int), but got {type(img_size)}')
        
        self.transform = transforms.Compose(
            [
                Rescale(self.img_size),
                RandomCrop(self.img_size),
                ToTensor()
            ]
        )
        
        self.classes = os.listdir(DATA_DIR)
        
        self.img_datasets = FoxDataset(
            root_dir=DATA_DIR,
            transform=self.transform
        )
        
        self.data_loader = DataLoader(
            self.img_datasets,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=3
        )
    
    def train_test_split_dl(
        self, 
        dataset, 
        train_size=None, 
        test_size=None,
        shuffle=True,
        num_workers=0
        ):
        """Splits data loader into training and testing dataloaders

        Args:
            dataset (torch.utils.data.Dataset): RGB image datasets 
            train_size (float, optional): size of training dataset. Defaults to None, in which case it is set to 0.8.
            test_size (float, optional): size of testing dataset. Defaults to None, in which case it is set to 0.2.
            shuffle (bool, optional): whether to shuffle the training sets. Defaults to True.
            num_workers (int): number of workers for the dataloader. Defaults to 0.

        Returns:
            tuple(torch.utils.DataLoader, torch.utils.DataLoader): tuple of training and testing dataloaders, in that order
        """
        if train_size and test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        elif train_size is not None and test_size is None:
            test_size = 1 - train_size
        elif train_size is None and test_size is not None:
            train_size = 1 - test_size
            
        if train_size + test_size != 1.0:
            raise ValueError(f'train_size and test_size must add up to 1, but instead adds up to {train_size + test_size}.')

        if train_size >= 1 or test_size >= 1:
            raise ValueError(f'train_size and test_size must both be less than 1')
        elif train_size <= 0 or test_size <= 0:
            raise ValueError(f'train_size and test_size must both be less than 0')
            
        train_data, test_data = random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return train_loader, test_loader
    
#  for testing

if __name__ == '__main__':
    transform = transforms.Compose(
        [
            Rescale(IMG_SIZE),
            RandomCrop(IMG_SIZE), 
            ToTensor()
        ]
            )
    dataset = FoxDataset(
        root_dir=DATA_DIR,
        transform=transform
    )
    
    for i, image in enumerate(dataset, 0):
        print(image[0].size(), image[1].size())
        
        if i == 3:
            break
       