import cv2
import torch
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
import os

# 이미지 로드
class dataLoad:
  
  # datasets
  path_images = r'/home/ubuntu/.Project/datasets/images/'
  path_masks = r'/home/ubuntu/.Project/datasets/masks/'  

  images_paths = glob.glob(path_images + '*.jpg')
  masks_paths = glob.glob(path_masks + '*.jpg')
  df = pd.DataFrame({'images': images_paths, 'masks': masks_paths})

# Noemalize
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

mask_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ])

# 데이터 로드
class Team21_MyDataset(torch.utils.data.Dataset):
  def __init__(self, df=dataLoad.df, image_transform=image_transform, mask_transform=mask_transform):
    self.df = df
    self.image_transform = image_transform
    self.mask_transform = mask_transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    image_path = self.df.loc[idx, 'images']
    mask_path = self.df.loc[idx, 'masks']

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)
    # mask =cv2.imread(mask_path, 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if self.image_transform:
      image = self.image_transform(image).float()

    if self.mask_transform:
      mask = self.mask_transform(mask)
    #   mask = mask.squeeze(0)
    return image, mask



def prepare_loaders(df = dataLoad.df, 
                    train_num= int(dataLoad.df.shape[0] * .7), 
                    test_num= int(dataLoad.df.shape[0] * .86), 
                    bs = 8):
   
    train = df[:train_num].reset_index(drop=True)
    valid = df[train_num:test_num].reset_index(drop=True)
    test = df[test_num:].reset_index(drop=True)

    train_ds = Team21_MyDataset(df = train)
    valid_ds = Team21_MyDataset(df = valid)
    test_ds = Team21_MyDataset(df = test)

    train_loader = DataLoader(train_ds, 
                              batch_size = bs, 
                              shuffle = True,) 
                              # pin_memory = True,
                              # num_workers = os.cpu_count(),
                              # drop_last = True)
    
    valid_loader = DataLoader(valid_ds, 
                              batch_size = bs, 
                              shuffle = True,)
                              # pin_memory = True,
                              # num_workers = os.cpu_count(),
                              # drop_last = True)
    
    test_loader = DataLoader(test_ds, 
                             batch_size = bs, 
                             shuffle = True,)
                            # pin_memory = True,
                            # num_workers = os.cpu_count(),
                            # drop_last = True)
    
    print("data loader finish")
    return train_loader, valid_loader, test_loader