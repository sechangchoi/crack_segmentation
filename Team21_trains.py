# argparse, main 불러오기, 
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
import segmentation_models_pytorch as smp

from Team21_models import Team21_Model
from Team21_datasets import *
# from Team21_datasets import prepare_loaders
from Team21_datasets import dataLoad as dl
from Team21_trains import * 

#  certificate verify failed: certificate has expired 오류에 대한 코드
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print(dl.df.head(2))

# argparse
def define():
    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, default = "unetplusplus")
    p.add_argument('--encoder', type=str, default = "resnet34")
    p.add_argument('--n_epochs', type = int, default = 100)

    config = p.parse_args()

    return config
# python --model "fpn"
def main(config):

    train_loader, valid_loader, test_loader = prepare_loaders(df = dl.df, 
                                                            train_num= int(dl.df.shape[0] * .7), 
                                                            test_num= int(dl.df.shape[0] * .86), 
                                                            bs = 8)

    # GPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Currently using "{device}" device.')

    # context = ssl._create_unverified_context()
    # response = urllib.request.urlopen(requests, data=data.encode('utf-8'), context=context)

    # Model
    # model= Team21_Model("Unet", "resnet34", in_channels=3, out_classes=1).to(device)

    if config.model == 'Unet':
        model= Team21_Model(arch = "Unet", encoder_name = config.encoder, #"resnet34", 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'Unet++':
        model= Team21_Model(arch = "UnetPlusPlus", encoder_name = config.encoder,
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'DeepLabV3':
        model= Team21_Model(arch = "DeepLabV3", encoder_name = config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'DeepLabV3Plus':
        model= Team21_Model(arch = "DeepLabV3Plus", encoder_name = config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'MAnet':
        model= Team21_Model(arch = "MAnet", encoder_name =config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'Linknet':
        model= Team21_Model(arch = "Linknet", encoder_name = config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'FPN':
        model= Team21_Model(arch = "FPN", encoder_name = config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'PSPNet':
        model= Team21_Model(arch = "PSPNet", encoder_name = config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)
    elif config.model == 'PAN':
        model= Team21_Model(arch = "PAN", encoder_name = config.encoder, 
                            in_channels=3, out_classes=1).to(device)
        print(model)


    # Wandb
    wandb.login()
    wandb_logger = WandbLogger(project='YearDream_Team21', entity="deepcrack")


    pl.seed_everything(2022)

    # Early Stopping (mode가 min이 맞나?? val이면 max아닌가??)
    early_stop = EarlyStopping(monitor="valid_loss", mode="min", patience = 40)

    # Model Checkpoint Setting
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_dataset_iou",
        filename= config.model + config.encoder + "_{epoch:02d}_{val/loss:.4f}",
        dirpath='/home/ubuntu/.Project/saved/' + config.encoder + '/',
        mode= 'max',
        save_top_k = 10,
        save_weights_only=True)

    trainer = pl.Trainer(
        logger = wandb_logger,
        gpus= -1, 
        max_epochs= config.n_epochs,
        callbacks=[checkpoint_callback, early_stop]
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader,
    )

if __name__ == '__main__':
    config = define()
    main(config)