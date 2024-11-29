from db import FoxDB
from models.trainer import Trainer

from utils.transforms import ToTensor, Rescale, RandomHorizontalFlip
from utils.const import DATA_DIR, IMG_SIZE

import torchvision.transforms as transforms


if __name__ == '__main__':
    transforms = transforms.Compose(
        [
            Rescale((IMG_SIZE, IMG_SIZE)),
            RandomHorizontalFlip(),
            ToTensor()
        ]
    )
    db = FoxDB(
        root_dir=DATA_DIR,
        transform=transforms
    )
    
    redo = False 
    
    if redo is True:
        db.drop_table()
        db.create_fox_train()
        db.insert_fox_train(verbose=True)
    
    if db.get_length() == 0:
        db.create_fox_train()
        db.insert_fox_train()
        
    # trainer = Trainer()
    
    # print(trainer.count_parameters(), trainer.count_neurons())
    
    # num_epochs = 15
    
    # trainer.train_over_epoch(num_epochs=num_epochs)
    