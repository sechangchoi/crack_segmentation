from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.models.segmentation import *

from Team21_models import Team21_Model

from utils import get_transform, visualize


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
print(f'Currently using "{device}" device.')

app = FastAPI()

architecture = 'Unetplusplus'
encoder = 'resnet34'


check_path = '/home/ubuntu/.Project/saved/resnet34/Unet++resnet34_epoch=177_val/loss=0.2046.ckpt'
model= Team21_Model(arch = architecture, encoder_name = encoder, in_channels=3, out_classes=1)
model = model.load_from_checkpoint(check_path, arch = architecture, encoder_name=encoder).to(device)
model.eval()

transform = get_transform()

class InputData(BaseModel):
    image: Any

@app.post("/predict")
async def predict(data: InputData):
    img = np.array(data.image, dtype=np.uint8)
    tensor_img = transform(img).unsqueeze(0).cuda()
    
    h, w = tensor_img.shape[2:]
    if h % 32 != 0:
        tensor_img = F.pad(tensor_img, (0, 0, (h//32+1) * 32 - h, 0) )
        
    if w % 32 != 0:
        tensor_img = F.pad(tensor_img, (0, (w//32+1) * 32 - w, 0, 0))

    output = model(tensor_img).squeeze(0).squeeze(0).detach().cpu().numpy()
    label_map = visualize(output)
    
    return JSONResponse({'segmentation' : label_map.tolist()})
