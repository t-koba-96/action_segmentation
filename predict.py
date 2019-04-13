import torch
import torch.nn as nn

import numpy as np
import datas
import os
import argparse

from models import network,regression
from utils import dataset,loader,test,util


"""default"""

MODEL='Twostream_TCN'
BATCH_SIZE=2
IMAGE_SIZE=224
CLIP_LENGTH=63
SLIDE_STRIDE=63
CLASSES=11
DEVICE='cuda:0'
WEIGHT_PATH=MODEL+'_result/finish'
RESULT_NAME=MODEL+'_b'
TRAIN_VIDEO_LIST=[2]


"""change parameters here"""

def get_arguments():

    parser = argparse.ArgumentParser(description='training action segmentation network')

    parser.add_argument("--model", type=str, default=MODEL,
                        help="available models => Attention_TCN/Twostream_TCN/Dual_Attention_TCN")
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
    parser.add_argument("--weight_path", type=str, default=WEIGHT_PATH,
                        help="path for saved weights, tensorboard")
    parser.add_argument("--result_name", type=str, default=RESULT_NAME,
                        help="file name for saving csv results")
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

     if args.model == 'Attention_TCN':
         net = network.attention_tcn(args.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()
         test.create_data_csv(testloader,net,args.device,args.classes,args.result_name,two_stream=False)
         test.create_demo_csv(testloader,net,args.device,classes,args.result_name,args.clip_length,two_stream=False)

     elif args.model == 'Twostream_TCN':
         net = network.twostream_tcn(args.classes)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()
         test.create_data_csv(testloader,net,args.device,args.classes,args.result_name,two_stream=True)
         test.create_demo_csv(testloader,net,args.device,classes,args.result_name,args.clip_length,two_stream=True)

     elif args.model == 'Dual_Attention_TCN':
         pose_net = regression.r_at_vgg(args.classes)
         pose_net=nn.DataParallel(pose_net)
         net = network.dual_attention_tcn(args.classes,pose_net)
         net = nn.DataParallel(net)
         net.load_state_dict(torch.load(os.path.join("weight","main",args.weight_path+".pth")))
         net = net.to(args.device)
         net.eval()
         test.create_data_csv(testloader,net,args.device,args.classes,args.result_name,two_stream=False)
         test.create_demo_csv(testloader,net,args.device,classes,args.result_name,args.clip_length,two_stream=False)


if __name__ == '__main__':
    main()