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
    parser.add_argument('video' , type=str, help='which video uisng for test, use alphabet(1=a,2=b,3=c ....)')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')
    parser.add_argument('--pose', default=False , help='show pose result or not')

    return parser.parse_args()



def main():

     args = get_arguments()
     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))
     device=torch.device(args.device)
     classes=datas.class_list()

     test_video_list=[]
     if args.video == 'a':
         test_video_list.append(1)
     elif args.video == 'b':
         test_video_list.append(2)
     elif args.video == 'c':
         test_video_list.append(3)
     elif args.video == 'd':
         test_video_list.append(4)
     elif args.video == 'e':
         test_video_list.append(5)
     video_path_list,_,__,label_path_list,pose_path_list = datas.test_path_list(test_video_list)

     frameloader = dataset.Video(video_path_list,label_path_list,pose_path_list,
                                 SETTING.image_size,SETTING.clip_length,
                                 SETTING.clip_length,SETTING.classes)
     testloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,
                                             shuffle=False,num_workers=SETTING.num_workers,
                                             collate_fn=loader.my_collate_fn)

     if SETTING.model == 'Attention_TCN':
         net = network.attention_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,two_stream=False)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=False)

     elif SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()
         test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,two_stream=True)
         test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=True)

     elif SETTING.model == 'Dual_Attention_TCN':
         if args.pose == False:
             pose_net = regression.r_at_vgg(SETTING.classes)
             pose_net=nn.DataParallel(pose_net)
             net = network.dual_attention_tcn(SETTING.classes,pose_net)
             net = nn.DataParallel(net)
             net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
             net = net.to(device)
             net.eval()
             test.create_data_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,two_stream=False)
             test.create_demo_csv(testloader,args.video,net,device,classes,SETTING.save_file,SETTING.clip_length,two_stream=False)
         else:
             if SETTING.reg_model == 'Attention_VGG':
                 net = regression.r_at_vgg(SETTING.classes)
                 net = nn.DataParallel(net)
                 net = net.to(device)
                 net.eval()
                 test.create_pose_csv(testloader,args.video,net,device,SETTING.classes,SETTING.save_file,two_stream=False)

if __name__ == '__main__':
    main()