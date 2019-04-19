from PIL import Image
import os
import argparse
import pandas as pd 
import numpy as np
import datas


def get_arguments():
    
    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('mode', type=str, help='train or test')
    parser.add_argument('cutout', type=int, help='cutout size')

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

def main():
     args = get_arguments()
     if args.mode == 'train':
         video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.train_path_list([1,2,3,4,5])
     elif args.mode == 'test':
         video_path_list,left_cutout_path_list,right_cutout_path_list,label_path_list,pose_path_list = datas.test_path_list([1,2,3,4,5])
    
     cut_out_image(args.cutout,video_path_list,left_cutout_path_list,right_cutout_path_list,pose_path_list)


if __name__ == '__main__':
    main()