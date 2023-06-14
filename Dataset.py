import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class FashionDataset(Dataset):
    def __init__(self, data, transform=None):        
        self.fashion = list(data.values)
        self.transform = transform
        
        label, image = [], []
        
        for i in self.fashion:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        
        if self.transform is not None:
            # transfrom the numpy array to PIL image before the transform function
            pil_image = Image.fromarray(np.uint8(image)) 
            image = self.transform(pil_image)
            
        return image, label


## Data Mnist Image Folder
train_transform1 = transforms.Compose([
    transforms.RandomChoice([
        transforms.Compose([transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    ]),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
    transforms.Resize((70,70)),
    transforms.ToTensor(),
    ])

train_transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((70,70)),
    transforms.ToTensor(),
    ])

train_transform3 = transforms.Compose([
    transforms.Resize((70,70)),
    transforms.ToTensor(),
    ])

train_transform4 = transforms.Compose([
    transforms.Resize((70,70)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=1, scale=(0.02, 0.2)),
    ])

test_transform = transforms.Compose([
    transforms.Resize((70,70)),
    transforms.ToTensor(),
    ])


train_csv = pd.read_csv(r"DatasetV2\train\Hexacore-Train.csv")
test_csv = pd.read_csv(r"DatasetV2\test\Hexacore-Test.csv")
## import data
train_set1 = FashionDataset(train_csv,transform=train_transform1)
train_set2 = FashionDataset(train_csv,transform=train_transform2)
train_set3 = FashionDataset(train_csv,transform=train_transform3)
train_set4 = FashionDataset(train_csv,transform=train_transform4)
train_sets = torch.utils.data.ConcatDataset([train_set1, train_set2, train_set3, train_set4])

test_set = FashionDataset(test_csv,transform=test_transform)

train_set, val_set = torch.utils.data.random_split(train_sets, [int(len(train_sets) * 0.9), int(len(train_sets) * 0.1)])

train_dataloader = torch.utils.data.DataLoader(train_set,
                                        batch_size = 100,
                                        shuffle = True)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                        batch_size = 100)

loaders = {
    'train': train_dataloader,
    'val': val_dataloader
}