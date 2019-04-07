from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import datas
import os
from models import cnn,network
from utils import dataset,loader,test,util

#batch size
batch_size=2

#image size
image_size=224

#clip_length
clip_length=63

#class number
class_num=11

#weight_file_name
file_name="tcn"

#csv_name
csv_name="working_a_test"

#gpu activate
device=torch.device('cuda:0')

#classes
classes=datas.class_list()

#worker_num(start,end)
video_path_list,label_path_list=datas.test_path_list([1])

#Video(videopathlist,labelpathlist,image_size,clip_length,slide_num)
frameloader=dataset.Video(video_path_list,label_path_list,image_size,clip_length,clip_length,class_num)

testloader=torch.utils.data.DataLoader(frameloader,batch_size=batch_size,shuffle=False,num_workers=2,collate_fn=loader.my_collate_fn)

#model
net = network.attention_tcn(class_num)
net=nn.DataParallel(net)
net.load_state_dict(torch.load(os.path.join("weight",file_name+".pth")))
net=net.to(device)
net.eval()


# make csv files 

test.create_data_csv(testloader,net,device,class_num,csv_name)

test.create_demo_csv(testloader,net,device,classes,csv_name,clip_length)