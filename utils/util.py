
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import cv2


def imshape(image):
    image=image/2+0.5
    npimg=image.numpy()
    return np.transpose(npimg,(1,2,0))


def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show


def one_hot_2d(labels,class_num):
    y = labels.view(labels.size(0),-1)
    y_onehot = torch.FloatTensor(labels.size(0),class_num)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
   
    return y_onehot


def one_hot_3d(labels,class_num):
    y = labels.view(labels.size(0),labels.size(1),-1)
    y_onehot = torch.FloatTensor(labels.size(0),labels.size(1),class_num)
    y_onehot.zero_()
    y_onehot.scatter_(2, y, 1)
   
    return y_onehot


def normalize_heatmap(x):
    min = x.min()
    max = x.max()
    result = (x-min)/(max-min)
    return result

def normalize_attention_conv(x,channel,feat):
    x=x.view(-1,channel,feat*feat)
    max,_=x.max(dim=2)
    max=max.view(-1,channel,1)
    min,_=x.min(dim=2)
    min=min.view(-1,channel,1)
    result = ((x-min)/(max-min)).view(-1,channel,feat,feat)
    
    return result
    