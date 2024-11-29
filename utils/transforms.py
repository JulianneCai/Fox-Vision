import numpy as np

import skimage
import torch

from PIL import Image


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
    

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image):
        if torch.rand(1) < self.p:
            image = image.transpose((1, 0, 2))
        return image
            