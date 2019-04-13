import torch
import torch.nn as nn

import numpy as np
import datas
import os
import argparse
import yaml

from addict import Dict
from models import network,regression
from utils import dataset,loader,test,util



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

     device=torch.device(args.device)

     classes=datas.class_list()

     video_path_list,label_path_list,pose_path_list = datas.test_path_list(SETTING.test_video_list)

     frameloader = dataset.Video(video_path_list,label_path_list,pose_path_list,SETTING.image_size,SETTING.clip_length,SETTING.clip_length,SETTING.classes)

     testloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,shuffle=False,num_workers=SETTING.num_workers,collate_fn=loader.my_collate_fn)

     if SETTING.model == 'Attention_TCN':
         net = network.attention_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,net,device,SETTING.classes,SETTING.save_file,two_stream=False)
         test.create_demo_csv(testloader,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=False)

     elif SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,net,device,SETTING.classes,SETTING.save_file,two_stream=True)
         test.create_demo_csv(testloader,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=True)

     elif SETTING.model == 'Dual_Attention_TCN':
         pose_net = regression.r_at_vgg(SETTING.classes)
         pose_net=nn.DataParallel(pose_net)
         net = network.dual_attention_tcn(SETTING.classes,pose_net)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,net,device,SETTING.classes,SETTING.save_file,two_stream=False)
         test.create_demo_csv(testloader,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=False)


if __name__ == '__main__':
    main()