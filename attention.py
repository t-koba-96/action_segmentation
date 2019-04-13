import torch 
import torch.nn as nn

import os
import random
import datas
import argparse
import matplotlib.pyplot as plt

from utils import util,dataset,loader,test
from models import regression,network


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


"""change parameters here"""

def get_arguments():

    parser = argparse.ArgumentParser(description='training action segmentation network')

    parser.add_argument("--model", type=str, default=MODEL,
                        help="available models => Attention_VGG,Attention_TCN,Twostream_TCN,Dual_Attention_TCN")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="image size")
    parser.add_argument("--clip_length", type=int, default=CLIP_LENGTH,
                        help="number of video frames clipped")
    parser.add_argument("--slide_stride", type=int, default=SLIDE_STRIDE,
                        help="slide stride through dataset")
    parser.add_argument("--classes", type=int, default=CLASSES,
                        help="number of classification classes")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="gpu device")
    parser.add_argument("--attention_path", type=str, default=ATTENTION_PATH,
                        help="attention weight path")
    parser.add_argument("--result_name", type=str, default=RESULT_NAME,
                        help="file name for saving weights, tensorboard")
    parser.add_argument("--train_list", type=list, default=TRAIN_VIDEO_LIST,
                        help="video list using for training")

    return parser.parse_args()



def main():

     args = get_arguments()

     device=torch.device(args.device)

     classes=datas.class_list()

     video_path_list,label_path_list,pose_path_list=datas.test_path_list(args.train_list)

     frameloader=dataset.Video(video_path_list,label_path_list,pose_path_list,args.image_size,args.clip_length,args.slide_stride,args.classes)

     testloader=torch.utils.data.DataLoader(frameloader,batch_size=args.batch_size,shuffle=False,num_workers=2,collate_fn=loader.my_collate_fn)

     if args.model == 'Attention_VGG':
         net = regression.r_at_vgg(args.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","reg",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()

     elif args.model == 'Attention_TCN':
         net = network.attention_tcn(args.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()

     elif args.model == 'Twostream_TCN':
         net = network.twostream_tcn(args.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()

     elif args.model == 'Dual_Attention_TCN':
         pose_net = regression.r_at_vgg(args.classes)
         pose_net=nn.DataParallel(pose_net)
         net = network.dual_attention_tcn(args.classes,pose_net)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()

     at_net = cnn.attention_net(net)
     at_net = at_net.to(device)
     at_net.eval()
     test.show_attention(testloader,at_net,device,save_name,two_stream=False)


if __name__ == '__main__':
    main()