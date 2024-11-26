import sqlite3
import io
import numpy as np
import torch
import glob
import sys

from collections import defaultdict

from utils.processImage import FoxDataset, ImageProcessor, Rescale, RandomCrop, ToTensor
from utils.const import DATA_DIR, IMG_SIZE, BATCH_SIZE

from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FoxDatabaseFormatting(Dataset):
    """ Collates information fox images stored locally for database insertion """
    def __init__(self):
        
        self.class_map = {'red-fox': 1, 'arctic-fox': 0}
        
        self.data = []
        
        self.transform = transforms.Compose(
            [
                Rescale(IMG_SIZE),
                RandomCrop(IMG_SIZE),
                ToTensor()
            ]
        )
    
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
        
    def __getitem__(self, key):
        if torch.is_tensor(key):
            key = key.tolist()
        img_path, class_name = self.data[key]
        
        #  dtype must be float32 otherwise Conv2d will complain
        rgb_mat = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        
        class_id = self.class_map[class_name] 
        
        rgb_mat = self.transform(rgb_mat)
            
        return class_id, class_name, rgb_mat


class FoxDB:
    """ Handles manipulation of fox database """
    def __init__(self):
        self.conn = sqlite3.connect(
            'foxes.db', 
            detect_types=sqlite3.PARSE_DECLTYPES,
            autocommit=True
            )
        self.cursor = self.conn.cursor()
        self.transform = transforms.Compose(
            [
                Rescale(IMG_SIZE),
                RandomCrop(IMG_SIZE),
                ToTensor()
            ]
        )
        
        self.dataset = FoxDatabaseFormatting()
       
        self.img_process = ImageProcessor(
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE
        )
        
    def _adapt_array(self, arr):
        """Use binary data-bytes to push non-text data into sqlite

        Args:
            arr (numpy.ndarray): numpy array of data

        Returns:
            sqlite.Binary: binary data
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def _convert_array(self, text):
        """Converts binary data back to numpy array
        
        Args:
            text (str): the text
            
        Returns:
            numpy.ndarray: array of converted data
        """
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    
    def create_fox_train(self):
        """Creates table if one does not already exist"""
        self.cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS Foxes (
            img_id TEXT PRIMARY KEY,
            class_id INTEGER,
            class_name TEXT,
            rgb_mat ARRAY
            );
        """
        )
        
    def drop_table(self):
        self.cursor.execute(
        """--sql
        DROP TABLE IF EXISTS Foxes;
        """
        )
        
    def insert_fox_train(self):
        """ Inserts binary fox images into SQL database """
        sqlite3.register_adapter(np.ndarray, self._adapt_array)
        
        #  counter for how many class_names we have
        #  will be used for img_id later
        class_counts = defaultdict(int)
        
        for sample in enumerate(self.dataset):
            #  need to do this since sqlite3 doesn't support storing Tensor objects
            class_id = sample[1][0]
            
            #  string representation of class
            class_name = sample[1][1]
            
            #  increment class counter for img_id
            class_counts[class_name] += 1
            
            #  same here, RGB matrix is Tensor object
            rgb_mat = sample[1][2].detach().numpy()
            
            img_id = class_name + '-' + str(class_counts[class_name])
            
            self.cursor.execute(
                """--sql
                INSERT INTO Foxes (img_id, class_id, class_name, rgb_mat) 
                VALUES (?, ?, ?, ?);
                """, (img_id, class_id, class_name, rgb_mat)
                )
            
    def read_fox_train(self):
        """ Reads fox data from SQL database, and then converts it back into a numpy array
        
        Returns:
            numpy.ndarray: fox dataset
        """
        sqlite3.register_converter('array', self._convert_array)
        
        self.cursor.execute(
        """--sql
        SELECT rgb_mat FROM Foxes;
        """
        )
        rows = self.cursor.fetchall()
        return rows


if __name__ == '__main__':
    db = FoxDB()
    
    db.create_fox_train()
    db.insert_fox_train()
    
    #  to test if the values were inserted properly
    values = db.read_fox_train()
    
    print(np.array(values).shape)
    