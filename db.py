import sqlite3

from utils.processImage import FoxDataset, ImageProcessor, DataLoader


class FoxDB:
    def __init__(self):
        self.conn = sqlite3.connect('database/foxes.db')
        self.cursor = self.conn.cursor()
        self.batch_size = 64
        
        im_process = ImageProcessor()
        im_process.load_images()
        train = im_process.get_train()
        
        trans_train = FoxDataset(X=train)
        
        self.train_imgset = DataLoader(
            trans_train, 
            self.batch_size, 
            shuffle=True, 
            num_workers=3, 
            pin_memory=True
            )
        
    def create_fox_train(self):
        self.cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS foxes (col1 INTEGER)
        """
        )
        
    def load_fox_train(self):
        for images in self.train_imgset:
            self.cursor.execute('INSERT INTO foxes (col1) VALUES (?)', images)
            
    def read_fox_train(self):
        self.cursor.execute('SELECT col1 FROM foxes')
        rows = self.cursor.fetchall()
        return rows
        

if __name__ == '__main__':
    db = FoxDB()
    db.create_fox_train()
    db.load_fox_train()
    values = db.read_fox_train()
    
    print(values) 