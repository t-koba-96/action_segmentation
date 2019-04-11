from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datas
from models import regression
from utils import util,dataset,loader,train


#batch size
batch_size=2

#image size
image_size=224

#clip_length
clip_length=63

#dataset_slide
dataset_slide=1

#epochs
num_epochs=3 

#class number
class_num=11

#learning rate
lr=0.0002

# beta1 hyperparam for adam
beta1=0.5

# save file name (tensorboard , weight)
file_name="reg"

#gpu activate
device=torch.device('cuda:0')

#worker_num(start,end)
video_path_list,label_path_list,pose_path_list=datas.train_path_list([1,3,4,5])

#Video(videopathlist,labelpathlist,image_size,clip_length,slide_num)
frameloader=dataset.Video(video_path_list,label_path_list,pose_path_list,image_size,clip_length,dataset_slide,class_num)

trainloader=torch.utils.data.DataLoader(frameloader,batch_size=batch_size,shuffle=True,num_workers=2,collate_fn=loader.my_collate_fn)

#network
net = regression.r_vgg()
net = nn.DataParallel(net)
net = net.to(device)

#cross  entropy
criterion=nn.MSELoss()

#adam
optimizer = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
#optimizer=optim.Adam(net.parameters(),lr=lr,betas=(beta1,0.999))

#training
train.regression_train(trainloader,net,criterion,optimizer,device,num_epochs,file_name)