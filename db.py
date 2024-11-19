import sqlite3
import io
import numpy as np

from utils.processImage import FoxDataset, ImageProcessor, DataLoader


class FoxDB:
    def __init__(self):
        self.conn = sqlite3.connect('foxes.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        self.batch_size = 64
        
        im_process = ImageProcessor()
        im_process.load_images()
        train = im_process.get_train()
        
        trans_train = FoxDataset(X=train)
        
        self.train_loader = DataLoader(
            trans_train, 
            self.batch_size, 
            shuffle=True, 
            num_workers=3, 
            pin_memory=True
            )
    def _adapt_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def _convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    
    def create_fox_train(self):
        self.cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS foxes (arr array)
        """
        )
        
    def insert_fox_train(self):
        sqlite3.register_adapter(np.ndarray, self._adapt_array)
        
        for images in self.train_loader:
            images = images.detach().numpy()
            self.cursor.execute('INSERT INTO foxes (arr) VALUES (?)', (images, ))
            
    def read_fox_train(self):
        sqlite3.register_converter('array', self._convert_array)
        
        self.cursor.execute('SELECT arr FROM foxes')
        rows = self.cursor.fetchall()
        return rows
        

if __name__ == '__main__':
    db = FoxDB()
    db.create_fox_train()
    db.insert_fox_train()
    
    #  to test if the values were inserted properly
    # values = db.read_fox_train()