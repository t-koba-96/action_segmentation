import torch
import torch.nn as nn
import pandas as pd 
import os
import numpy as np
from . import util
import cv2
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

def accuracy(testloader,net,device,csv_path):
   correct=0
   total=0
   with torch.no_grad():
      for data in testloader:
         images,targets,labels,poses=data
         images = images.to(device)
         outputs=net(images)
         outputs=outputs.cpu()
         outputs=nn.Softmax(dim=1)(outputs)
         _,predicted=torch.max(outputs,1)
         labels=labels.view(-1)
         total += labels.size(0)
         correct += (predicted == labels).sum().item()

   return total,correct


def create_data_csv(testloader,net,device,class_num,csv_name):
   classes=[]
   correct_=[]
   total_=[]
   accuracy=[]   
   correct=torch.zeros(class_num+1).numpy()
   total=torch.zeros(class_num+1).numpy()
   correct.dtype='int32'
   total.dtype='int32'
   np.set_printoptions(precision=2)

   with torch.no_grad():
      for data in testloader:
         images,targets,labels,poses=data
         images = images.to(device)
         outputs=net(images)
         outputs=outputs.cpu()
         outputs=nn.Softmax(dim=1)(outputs)
         _,predicted=torch.max(outputs,1)
         labels=labels.view(-1)
         predicted_np=predicted.numpy()
         for i in range(labels.size(0)):
             total[labels[i]] += 1
             total[class_num] += 1
             if predicted[i] == labels[i]:
                correct[predicted_np[i]] += 1
                correct[class_num] += 1

   for i in range(class_num):
      classes.append(i)
      correct_.append(correct[i])
      total_.append(total[i])
      accuracy.append(correct[i]/total[i]*100)
   
   classes.append("total")
   correct_.append(correct[class_num])
   total_.append(total[class_num])
   accuracy.append(correct[class_num]/total[class_num]*100) 

   df = pd.DataFrame({
                    'classes' : classes,
                    'correct' : correct_,
                    'total' : total_,
                    'accuracy' : accuracy
   })

   df.to_csv(os.path.join("result","data",csv_name+".csv"))



def create_demo_csv(testloader,net,device,classes,csv_name,clip_length):
   num=[]
   Frame=[]
   c_s_l=[]
   c_l_n=[]
   c_s_t=[]
   c_t_n=[]
   t_s=[]
   n_=[]
   tp_=[]
   al_=[]
   ac_=[]
   ca_=[]
   gd_=[]
   as_=[]
   hp_=[]
   st_=[]
   ch_=[]
   co_=[]
   
   
   for i in range(2400):
      num.append(i+1)
      Frame.append("%s.png" % str(i).zfill(5))

   with torch.no_grad():
      for i,data in enumerate(testloader):
         images,targets,labels,poses=data
         images = images.to(device)
         outputs=net(images)
         outputs=outputs.cpu()
         outputs=nn.Softmax(dim=1)(outputs)
         best_score,predicted=torch.max(outputs,1)
         labels=labels.view(-1)
         frame_num=labels.size(0)
         labels_np=labels.numpy()
         predicted_np=predicted.numpy()
         best_score_np=best_score.numpy()
         outputs_np=outputs.numpy()
         for x in range(frame_num):
            c_s_l.append(classes[labels_np[x]])
            c_l_n.append(labels_np[x])
            c_s_t.append(classes[predicted_np[x]])
            c_t_n.append(predicted_np[x])
            t_s.append(best_score_np[x])
            n_.append(outputs_np[x,0])
            tp_.append(outputs_np[x,1])
            al_.append(outputs_np[x,2])
            ac_.append(outputs_np[x,3])
            ca_.append(outputs_np[x,4])
            gd_.append(outputs_np[x,5])
            as_.append(outputs_np[x,6])
            hp_.append(outputs_np[x,7])
            st_.append(outputs_np[x,8])
            ch_.append(outputs_np[x,9])
            co_.append(outputs_np[x,10]) 

   for x in range(2400%clip_length):
            c_s_l.append(classes[0])
            c_l_n.append(0)
            c_s_t.append(classes[0])
            c_t_n.append(0)
            t_s.append(0)
            n_.append(0)
            tp_.append(0)
            al_.append(0)
            ac_.append(0)
            ca_.append(0)
            gd_.append(0)
            as_.append(0)
            hp_.append(0)
            st_.append(0)
            ch_.append(0)
            co_.append(0)
   
         
   df = pd.DataFrame({
                    'number' : num,
                    'Frames' : Frame,
                    'class_str_label' : c_s_l,
                    'class_label_num' : c_l_n,
                    'class_str_top1' : c_s_t,
                    'class_top1_num' : c_t_n,
                    'Top1_score' : t_s,
                    'n' : n_,
                    'tp' : tp_,
                    'al' : al_,
                    'ac' : ac_,
                    'ca' : ca_,
                    'gd' : gd_,
                    'as' : as_,
                    'hp' : hp_,
                    'st' : st_,
                    'ch' : ch_,
                    'co' : co_
   })

   df.to_csv(os.path.join("result","demo",csv_name+".csv"))

def show_attention(testloader,net,device,save_name):

   with torch.no_grad():
      for i,data in enumerate(testloader):
         images,targets,labels,poses=data
         images_gpu = images.to(device)
         outputs=net(images_gpu)
         outputs=outputs.cpu()
         attention=outputs.detach()
         f_num=images.size(0)*images.size(1)*i
         images=images.view(-1,3,images.size(3),images.size(4))
         for x in range(images.size(0)):
              img=util.imshape(images[x,:,:,:])
              heatmap = attention[x,:,:,:]
              make_attention_map(img,heatmap,f_num,save_name)
              f_num+=1


def make_attention_map(img,heatmap,f_num,save_name):
    #attention map
    heatmap = heatmap.numpy()
    heatmap = np.average(heatmap,axis=0)
    heatmap = util.normalize_heatmap(heatmap)
    # 元の画像と同じサイズになるようにヒートマップのサイズを変更
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    #特徴ベクトルを256スケール化
    heatmap = np.uint8(255 * heatmap)
    # RGBに変更
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #戻す
    heatmap=heatmap/255
    # 0.5はヒートマップの強度係数
    s_img = heatmap * 0.5 + img
    #plt
    fig=plt.figure(figsize=(16,10))
    gs = GridSpec(4,5,left=0.13,right=0.9) 
    gs.update(wspace=-0.01)
    gs_1 = GridSpecFromSubplotSpec(nrows=4, ncols=3, subplot_spec=gs[0:4, 0:3])
    fig.add_subplot(gs_1[:, :])
    util.delete_line()
    plt.imshow(s_img)
    gs_2 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[0:2, 3:5])
    fig.add_subplot(gs_2[:,:])
    util.delete_line()
    plt.imshow(img)
    gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs[2:4, 3:5])
    fig.add_subplot(gs_3[:,:])
    util.delete_line()
    plt.imshow(heatmap)

    # Make the directory if it doesn't exist.
    SAVE_PATH = "../../../demo/images/attention"
    if not os.path.exists(os.path.join(SAVE_PATH,save_name)):
        os.makedirs(os.path.join(SAVE_PATH,save_name))
      
    plt.savefig(os.path.join(SAVE_PATH,save_name,str(f_num).zfill(5)+".png"))
    plt.close()
   
 