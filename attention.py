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
from models import regression,network,cnn


"""default"""

MODEL='Attention_VGG'
BATCH_SIZE=2
IMAGE_SIZE=224
CLIP_LENGTH=63
SLIDE_STRIDE=63
CLASSES=11
DEVICE='cuda:0'
ATTENTION_PATH='Attention_VGG_result/finish'
RESULT_NAME=MODEL+'_result'
TRAIN_VIDEO_LIST=[1,3,4,5]



'''
default == using 'cuda:0'
'''

def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')
    parser.add_argument('--pose', default=False , help='show pose_attention or not')

    return parser.parse_args()



def main():

     args = get_arguments()

     SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'))))

     device=torch.device(args.device)

     video_path_list,label_path_list,pose_path_list = datas.test_path_list(SETTING.test_video_list)

     frameloader = dataset.Video(video_path_list,label_path_list,pose_path_list,SETTING.image_size,SETTING.clip_length,SETTING.clip_length,SETTING.classes)

     testloader = torch.utils.data.DataLoader(frameloader,batch_size=SETTING.batch_size,shuffle=False,num_workers=SETTING.num_workers,collate_fn=loader.my_collate_fn)

     if SETTING.model == 'Attention_VGG':
         net = regression.r_at_vgg(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","reg",SETTING.save_file,SETTING.reg_batch+".pth")))
         net = net.to(device)
         net.eval()

     elif SETTING.model == 'Attention_TCN':
         if args.pose == False:
             net = network.attention_tcn(SETTING.classes)
             net = nn.DataParallel(net)
             net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
             net = net.to(device)
             net.eval()
         else:
             net = regression.r_at_vgg(SETTING.classes)
             net = nn.DataParallel(net)
             net.load_state_dict(torch.load(os.path.join("weight","reg",SETTING.save_file,SETTING.reg_batch+".pth")))
             net = net.to(device)
             net.eval()
         

     elif SETTING.model == 'Twostream_TCN':
         net = network.twostream_tcn(SETTING.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()

     elif SETTING.model == 'Dual_Attention_TCN':
         pose_net = regression.r_at_vgg(SETTING.classes)
         pose_net=nn.DataParallel(pose_net)
         net = network.dual_attention_tcn(SETTING.classes,pose_net)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",SETTING.save_file,SETTING.main_batch+".pth")))
         net = net.to(device)
         net.eval()

     at_net = cnn.attention_net(net)
     at_net = at_net.to(device)
     at_net.eval()
     test.show_attention(testloader,at_net,device,SETTING.save_file,two_stream=False)


if __name__ == '__main__':
    main()