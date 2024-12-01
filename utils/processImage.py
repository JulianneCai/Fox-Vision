import os

from typing import Tuple

import torch    
from torch.utils.data import Dataset, DataLoader, random_split

import db

       
class FoxDataset(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        """Iterative wrapper that allows PyTorch to iterate through 
        image, class_id during training.
        
        The class retrieves the image and class_id from a SQL database, which 
        is managed by the FoxDB class (see db.py), which contains methods that 
        send SQL queries to the database.

        Args:
            root_dir (str): root directory 
            transform (torchvision.transforms, optional): transformations to be applied. 
            Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.db = db.FoxDB(
            root_dir=self.root_dir,
            transform=self.transform
        )
       
    def __len__(self) -> int:
        return self.db.get_length()
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.db.retrieve_matrix(idx)
        class_id = self.db.retrieve_class_id(idx)
        
        image, class_id = self.db.to_tensor(image, class_id)
            
        return image, class_id

        
class ImageProcessor:
    """ Decomposes an image into a matrix of RGB values at each pixel 
    
    Args:
        root_dir (str): the directory containing the data files
        batch_size (int): size of batches
        img_size (int or tuple): size of image
        transform (torchvision.transforms): image transformations
    """
    def __init__(
            self, 
            root_dir, 
            batch_size, 
            img_size, 
            transform=None
        ) -> None:
        self.root_dir = root_dir
        self.batch_size = batch_size
        
        if isinstance(img_size, int) or isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise ValueError(f'expected img_size to be of type int or tuple(int, int), but got {type(img_size)}')
        
        self.transform = transform
        
        self.classes = os.listdir(self.root_dir)
        
        self.img_datasets = FoxDataset(
            root_dir=self.root_dir,
            transform=self.transform
        )
    
    def train_test_split_dl(
        self, 
        dataset, 
        train_size=None, 
        test_size=None,
        shuffle=True,
        num_workers=0
        ) -> Tuple[DataLoader, DataLoader]:
        """Splits data loader into training and testing dataloaders

        Args:
            dataset (torch.utils.data.Dataset): RGB image datasets 
            train_size (float, optional): size of training dataset. Defaults to None, in which case it is set to 0.8.
            test_size (float, optional): size of testing dataset. Defaults to None, in which case it is set to 0.2.
            shuffle (bool, optional): whether to shuffle the training sets. Defaults to True.
            num_workers (int): number of workers for the dataloader. Defaults to 0.

        Returns:
            Tuple[torch.utils.DataLoader, torch.utils.DataLoader]: tuple of training and testing dataloaders, in that order
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
    