#import
from __future__ import print_function
import os
import random
import torch
import datas
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import util,dataset,loader,test
from models import cnn,network


#batch size
batch_size=2

#image size
image_size=224

#clip_length
clip_length=63

#class number
class_num=11

#weight_file_name
file_name="tcn_1at"

#csv_name
save_name="attention_a_1at"

#gpu activate
device=torch.device('cuda:0')

#classes
classes=datas.class_list()

#worker_num(start,end)
video_path_list,label_path_list,pose_path_list=datas.test_path_list([1])

#Video(videopathlist,labelpathlist,image_size,clip_length,slide_num)
frameloader=dataset.Video(video_path_list,label_path_list,pose_path_list,image_size,clip_length,clip_length,class_num)

testloader=torch.utils.data.DataLoader(frameloader,batch_size=batch_size,shuffle=False,num_workers=2,collate_fn=loader.my_collate_fn)

#attention model
net = network.attention_tcn(class_num)
net=nn.DataParallel(net)
net.load_state_dict(torch.load(os.path.join("weight",file_name+".pth")))
net.eval()

at_net = cnn.attention_net(net)
at_net = at_net.to(device)
at_net.eval()

test.show_attention(testloader,at_net,device,save_name,two_stream=False)