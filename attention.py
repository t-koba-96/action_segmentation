import torch 
import torch.nn as nn

import os
import random
import datas
import yaml
import argparse
import matplotlib.pyplot as plt

from addict import Dict
from utils import util,dataset,loader,test
from models import network,cnn



'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('video' , type=str, help='which video uisng for test, use alphabet(1=a,2=b,3=c ....)')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    return parser.parse_args()



def main():

     args = get_arguments()
     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))
     device=torch.device(args.device)

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
                                 SETTING.clip_length,SETTING.classes,pose_label=True)
                                 
     testloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,
                                                 shuffle=False,num_workers=SETTING.num_workers,
                                                 collate_fn=loader.my_collate_fn)

     if SETTING.model == 'Attention_TCN':
         net = network.attention_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net.eval()

     elif SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net.eval()

     at_net = cnn.attention_net(net)
     at_net = at_net.to(device)
     at_net.eval()
     test.show_attention(testloader,args.video,at_net,device,SETTING.save_file,two_stream=False)


if __name__ == '__main__':
    main()