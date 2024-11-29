import sqlite3
import io
import numpy as np
import torch
import glob
import sys
import os

from collections import defaultdict

from utils.transforms import Rescale, ToTensor

from sklearn.preprocessing import LabelEncoder 

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FoxDatabaseFormatting(Dataset):
    """ Iterator wrapper for fox image information so that it can be 
    used for database insertion. 
    
    The main attributes here are class_id, which is the label-encoded 
    representation of the class name, class_name, which is the actual 
    class name, and rgb_mat, which is the transformed version of the 
    RGB matrix.
    
    Args: 
        root_dir (str): directory containing the training images
        transform (torchvision.transform.)
    """
    def __init__(self, root_dir, transform=None):
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
        
    def get_class_map(self):
        """Returns a dictionary of the class names, and the 
        encoded labels

        Returns:
            dict<str: int>: dict with key value given by class name, 
            and value given by encoded value 
        """
        return self.class_map
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path, class_name = self.data[idx]
        
        #  dtype must be float32 otherwise Conv2d will complain
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        
        class_id = self.class_map[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_id


class FoxDB:
    """ Handles manipulation of fox database. Has methods 
    for constructing a table in the database if one does not exist.
    The entries are given by a class_id, and the RGB matrix of the image.
    Both are turned into blobs and then inserted into the SQL database,
    since sqlite3 does not support arrays.
    
    There are also methods for getting the length of the database,
    inserting elements, and also retrieving elements."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.conn = sqlite3.connect(
            'foxes.db', 
            detect_types=sqlite3.PARSE_DECLTYPES,
            autocommit=True
            )
        self.cursor = self.conn.cursor()
        
        self.dataset = FoxDatabaseFormatting(
            root_dir=self.root_dir,
            transform=self.transform
        )
        
    def _adapt_array(self, arr):
        """Saves the array to a binary file in NumPy .npy format, 
        and then reads the bytes, and then stores them as a 
        sqlite3 binary data type so that it can be written into the 
        SQL database.

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
        """Reads the bytes stored in the SQL database, and then converts 
        them back into a numpy array.
        
        Args:
            text (str): the text
            
        Returns:
            numpy.ndarray: array of converted data
        """
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    
    def create_fox_train(self):
        """ Creates table if one does not already exist. The table contains
        the label-encoded name of the feature (class_id), and the RGB matrix 
        (rgb_mat)
        
        The class_id is the one-hot encoding of the class names. It will not 
        be viewable from DB browser since it is an array and has to be stored 
        as a blob. 
        
        Similarly for rgb_mat, which is the RGB matrix of the image,
        this is a numpy.ndarray object of dtype float32, and thus will also be 
        stored as a blob in the DB browser. """
        
        self.cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS Foxes (
                img_id TEXT PRIMARY KEY,
                rgb_mat ARRAY,
                class_id ARRAY
                )
            """
        )
        
    def drop_table(self):
        """ Deletes the table """
        self.cursor.execute(
            """--sql
            DROP TABLE IF EXISTS Foxes;
            """
        )
        
    def insert_fox_train(self, verbose=False):
        """ Inserts entire fox training dataset into SQL database 
        by converting the RGB matrices into binary data, and then 
        using our custom function to push non-text data into the 
        database 
        
        Args:
            verbose (bool): whether or not to have logging messages
        
        """
        
        #  registers np.ndarray as a type that can be inserted into the database
        sqlite3.register_adapter(np.ndarray, self._adapt_array)
        
        #  counter for how many class_names we have
        #  will be used for img_id later
        class_counts = defaultdict(int)
        
        class_map = self.dataset.get_class_map()
        
        inv_class_map = {value : key for key, value in class_map.items()}
        
        for _, (input, label) in enumerate(self.dataset, 0):
            #  convert torch.Tensor object to numpy.ndarray object 
            #  so that it can be inserted and selected from SQL database using 
            #  our adapters and converters
            input = input.detach().numpy()
            label = label.detach().numpy()
            
            class_counts[int(label)] += 1
            
            #  the img_id primary key is given by the class_name plus the instance number
            img_id = inv_class_map[int(label)] + '-' + str(class_counts[int(label)])
            
            if verbose is True:
                print(f'Inserting {img_id}')
            
            self.cursor.execute(
                """--sql
                INSERT INTO Foxes (img_id, rgb_mat, class_id) 
                VALUES (?, ?, ?);
                """, (img_id, input, label)
            )
    
    def get_cols(self):
        """Returns the columns of the SQL database table.
        
        Returns:
            list<str>: column names
        """
        self.cursor.execute(
            """--sql
            PRAGMA table_info(Foxes);
            """
        )
        col_names = self.cursor.fetchall()
        return col_names
    
    def retrieve_matrix(self, idx):
        """Retrieves the RGB matrix from the database. The selection assumes that 
        the primary key has the form class_name-n, where class_name is the name of 
        the class (e.g. arctic-fox, red-fox), and n is a unique integer.
        
        Args:
            idx (int): the row index
            
        Returns:
            numpy.ndarray: the RGB matrix. Should be of size (3, IMG_SIZE, IMG_SIZE).
            Note that the dimensions are permuted, and colour is the first dimension
            since that is what is needed for PyTorch
        """
        sqlite3.register_converter('ARRAY', self._convert_array)
        
        self.cursor.execute(
            """--sql
            SELECT rgb_mat FROM Foxes LIMIT 1 OFFSET (?)
            """, 
            (idx, )
        )
        
        row = self.cursor.fetchall()
        
        return row[0][0]
    
    def retrieve_matrix_by_class(self, idx, class_name):
        if class_name not in self.dataset.classes:
            raise ValueError(f'class {class_name} needs to be one of {self.dataset.classes}')
        
        sqlite3.register_converter('ARRAY', self._convert_array)
        
        self.cursor.execute(
            """--sql
            SELECT rgb_mat FROM Foxes
            WHERE img_id LIKE (?)
            LIMIT 1 OFFSET (?)
            """, 
            (f'{class_name}-%', idx)
        )
        
        row = self.cursor.fetchall()
        
        return row[0][0]
    
    def retrieve_class_id(self, idx):
        """Retrieves the label-encoded value of the class from the database
        
        Args:
            idx (int): the row index
            
        Returns:
            int: the label-encoded class
        """
        sqlite3.register_converter('ARRAY', self._convert_array)
        
        self.cursor.execute(
            """--sql
            SELECT class_id FROM Foxes LIMIT 1 OFFSET (?)
            """, 
            (idx, )
        )
            
        row = self.cursor.fetchall()
            
        return row[0][0]
    
    def retrieve_class_id_by_class(self, idx, class_name):
        """Retrives the idx'th row of class_name

        Args:
            idx (int): index 
            class_name (str): the class name. Must be one of the classes.

        Raises:
            ValueError: when the class name does not match up with an existing class
        """
        if class_name not in self.dataset.classes:
            raise ValueError(f'class {class_name} is not one of {self.dataset.classes}')
        sqlite3.register_converter('ARRAY', self._convert_array)
        
        self.cursor.execute(
            """--sql
            SELECT class_id FROM Foxes
            WHERE img_id LIKE (?)
            LIMIT 1 OFFSET (?)
            """, 
            (f'{class_name}-%', idx, )
        )
        
        row = self.cursor.fetchall()
        
        return row[0][0]
    
    def get_length(self):
        """Returns the length of SQL database
        
        Returns:
            int: length of SQL database
        """
        self.cursor.execute(
            """--sql
            SELECT COUNT(1) FROM Foxes
            """
        )
            
        count = self.cursor.fetchall()
        #  it gets stored as [(n, )] for some reason
        return count[0][0]
    
    def to_tensor(self, input, label):
        input = np.array(input, dtype=np.float32)
        input = torch.from_numpy(input)
        
        label = torch.tensor(label)
        
        return input, label


if __name__ == '__main__':
    from utils.const import DATA_DIR, IMG_SIZE
    from utils.processImage import FoxDataset
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose(
        [
            Rescale((IMG_SIZE, IMG_SIZE)),
            ToTensor()
        ]
            )
    
    db = FoxDB(root_dir=DATA_DIR, transform=transform)
    
    value = db.retrieve_matrix_by_class(idx=3, class_name='fox-girl')
    key = db.retrieve_class_id_by_class(idx=3, class_name='fox-girl')
    
    dataset = FoxDataset(
        root_dir=DATA_DIR,
        transform=transform
    )
    
    dl = DataLoader(
        dataset,
        shuffle=True,
        batch_size=4
    )
    
    for i, image in enumerate(dl, 0):
        print(image[0].size(), image[1].size())
        
        if i == 3:
            break
           