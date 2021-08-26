import torch
import torch.nn as nn



def model_param(model):
    param=0
    for i in model.parameters():
        param +=i.numel()
    return param
    
    
def model_init_kaim(model):
    for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                torch.nn.init.kaiming_uniform_(module.weight,mode='fan_in', nonlinearity='relu')


        