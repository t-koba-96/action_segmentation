from PIL import Image
import os
import torch
import argparse
import pandas as pd 
import numpy as np
import datas
from utils import util,dataset


def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('mode', type=str, help='train , test , demo , posemap')
    parser.add_argument('crop', type=int, help='crop size')

    return parser.parse_args()

def cut_out_image(cutout,video_path_list,left_cutout_path_list,right_cutout_path_list,pose_path_list):
    for num,v_path in enumerate(video_path_list):
        df = pd.read_csv(pose_path_list[num])
        dfa = np.array(df)
        if not os.path.exists(left_cutout_path_list[num]):
             os.makedirs(left_cutout_path_list[num])
        if not os.path.exists(right_cutout_path_list[num]):
             os.makedirs(right_cutout_path_list[num])
        for i in range(dfa.shape[0]):
             im = Image.open(os.path.join(v_path,str(i).zfill(5)+".png"))
             if num == 4:
                im.crop((dfa[i,0]-cutout*2, dfa[i,1]-cutout*2, dfa[i,0]+cutout*2, dfa[i,1]+cutout*2)).save(os.path.join(left_cutout_path_list[num],str(i).zfill(5)+".png"), quality=95)
                im.crop((dfa[i,2]-cutout*2, dfa[i,3]-cutout*2, dfa[i,2]+cutout*2, dfa[i,3]+cutout*2)).save(os.path.join(right_cutout_path_list[num],str(i).zfill(5)+".png"), quality=95)
             else:
                im.crop((dfa[i,0]/2-cutout, dfa[i,1]/2-cutout, dfa[i,0]/2+cutout, dfa[i,1]/2+cutout)).save(os.path.join(left_cutout_path_list[num],str(i).zfill(5)+".png"), quality=95)
                im.crop((dfa[i,2]/2-cutout, dfa[i,3]/2-cutout, dfa[i,2]/2+cutout, dfa[i,3]/2+cutout)).save(os.path.join(right_cutout_path_list[num],str(i).zfill(5)+".png"), quality=95)


def show_hand_image(cutout,testloader,video_num):
     for i,data in enumerate(testloader):
           images,left_img,right_img,targets,labels,poses=data
           images=images.view(-1,3,images.size(3),images.size(4))
           left_img=left_img.view(-1,3,left_img.size(3),left_img.size(4))
           right_img=right_img.view(-1,3,right_img.size(3),right_img.size(4))
           poses=poses.view(-1,poses.size(2))     
           f_num=images.size(0)*i
           for x in range(images.size(0)):
                img=util.imshape(images[x,:,:,:])
                l_img=util.imshape(left_img[x,:,:,:])
                r_img=util.imshape(right_img[x,:,:,:])
                pose=poses[x,:]
                util.make_hand_image(img,l_img,r_img,pose,video_num,f_num,cutout)
                f_num+=1



def main():
     args = get_arguments()
     if args.mode == 'train':
           video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.train_path_list([1,2,3,4,5])
           cut_out_image(args.crop,video_path_list,left_cutout_path_list,right_cutout_path_list,pose_path_list)
     elif args.mode == 'test':
           video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.test_path_list([1,2,3,4,5])
           cut_out_image(args.crop,video_path_list,left_cutout_path_list,right_cutout_path_list,pose_path_list)
     elif args.mode == 'demo':
          list=['a','b','c','d','e']
          for i,video_num in enumerate(list):
                video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.test_path_list([i+1])
                frameloader = dataset.Video(video_path_list,left_cutout_path_list,right_cutout_path_list,
                                              label_path_list,pose_path_list,
                                               224,60,60,11,cutout_img=True,pose_label=True)
                testloader = torch.utils.data.DataLoader(frameloader,batch_size=2,
                                                     shuffle=False,num_workers=2)
                show_hand_image(args.crop,testloader,video_num)
     elif args.mode == 'posemap':
          list=['a','b','c','d','e']
          if args.crop >= 200:
               gauss_size=251
          else:
               gauss_size=31
          for i,video_num in enumerate(list):
                video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.test_path_list([i+1])
                frameloader = dataset.Video(video_path_list,left_cutout_path_list,right_cutout_path_list,
                                              label_path_list,pose_path_list,
                                               224,60,60,11,pose_label=True)
                testloader = torch.utils.data.DataLoader(frameloader,batch_size=2,
                                                     shuffle=False,num_workers=2)
                util.show_posemap(testloader,video_num,224,gauss_size)

     


if __name__ == '__main__':
    main()