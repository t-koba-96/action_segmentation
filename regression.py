import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import datas
import argparse

from models import regression
from utils import util,dataset,loader,train


"""default"""

MODEL='Attention_VGG'
BATCH_SIZE=2
IMAGE_SIZE=224
CLIP_LENGTH=63
SLIDE_STRIDE=10
EPOCH=3
CLASSES=11
LEARNING_RATE=0.0002
BETA1=0.5
DEVICE='cuda:0'
RESULT_NAME=MODEL+'_result'
TRAIN_VIDEO_LIST=[1,3,4,5]


"""change parameters here"""

def get__arguments():

    parser = argparse.ArgumentParser(description='training action segmentation network')

    parser.add_argument("--model", type=str, default=MODEL,
                        help="available models => Attention_VGG")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="image size")
    parser.add_argument("--clip_length", type=int, default=CLIP_LENGTH,
                        help="number of video frames clipped")
    parser.add_argument("--slide_stride", type=int, default=SLIDE_STRIDE,
                        help="slide stride through dataset")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="training epoch")
    parser.add_argument("--classes", type=int, default=CLASSES,
                        help="number of classification classes")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=BETA1,
                        help="hyperparam for adam")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="gpu device")
    parser.add_argument("--result_name", type=str, default=RESULT_NAME,
                        help="file name for saving weights, tensorboard")
    parser.add_argument("--train_list", type=list, default=TRAIN_VIDEO_LIST,
                        help="video list using for training")

    return parser.parse_args()

def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    return parser.parse_args()



def main():

     args = get_arguments()

     device=torch.device(args.device)

     video_path_list,label_path_list,pose_path_list=datas.train_path_list(args.train_list)

     frameloader=dataset.Video(video_path_list,label_path_list,pose_path_list,args.image_size,args.clip_length,args.slide_stride,args.classes)

     trainloader=torch.utils.data.DataLoader(frameloader,batch_size=args.batch_size,shuffle=True,num_workers=2,collate_fn=loader.my_collate_fn)

     criterion=nn.MSELoss()

     if args.model == 'Attention_VGG':
         net = regression.r_at_vgg(args.classes)
         net = nn.DataParallel(net)
         net = net.to(args.device)
         optimizer=optim.Adam(net.parameters(),lr=args.lr,betas=(args.beta1,0.999))
         train.regression_train(trainloader,net,criterion,optimizer,args.device,args.epoch,args.result_name)


if __name__ == '__main__':
    main()