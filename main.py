import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import datas
import argparse
import yaml

from addict import Dict
from models import network,regression
from utils import util,dataset,loader,train


'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    return parser.parse_args()



def main():

     args = get_arguments()

     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))

     device = torch.device(args.device)

     video_path_list,label_path_list,pose_path_list = datas.train_path_list(SETTING.train_video_list)

     frameloader = dataset.Video(video_path_list,label_path_list,pose_path_list,SETTING.image_size,SETTING.clip_length,SETTING.slide_stride,SETTING.classes)

     trainloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,shuffle=True,num_workers=SETTING.num_workers,collate_fn=loader.my_collate_fn)

     criterion=nn.CrossEntropyLoss()

     if SETTING.model == 'Attention_TCN':
         net = network.attention_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,net,criterion,optimizer,device,SETTING.epoch,SETTING.save_file,two_stream=False)


     elif SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,net,criterion,optimizer,device,SETTING.epoch,SETTING.save_file,two_stream=True)

     elif SETTING.model == 'Dual_Attention_TCN':
         at_net = regression.r_at_vgg(SETTING.classes)
         at_net=nn.DataParallel(at_net)
         at_net.load_state_dict(torch.load(os.path.join("weight","reg",SETTING.save_file,SETTING.reg_batch+".pth")))
         net = network.dual_attention_tcn(SETTING.classes,at_net)
         net = nn.DataParallel(net)
         net = net.to(device)
         optimizer=optim.Adam(net.parameters(),lr=SETTING.learning_rate,betas=(SETTING.beta1,0.999))
         train.model_train(trainloader,net,criterion,optimizer,device,SETTING.epoch,SETTING.save_file,two_stream=False)


if __name__ == '__main__':
    main()