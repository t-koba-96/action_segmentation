import torch
import torch.nn as nn
import pandas as pd 
import os
import numpy as np
from . import util
import cv2
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def accuracy(testloader,net,device,csv_path):
   correct=0
   total=0
   with torch.no_grad():
      for data in testloader:
         images,targets,labels=data
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
         images,targets,labels=data
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




def create_demo_csv(testloader,net,device,classes,csv_name):
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
         images,targets,labels=data
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

def show_attention(images,net,device,save_name):
   images_gpu = images.to(device)    
   at_outputs=net(images_gpu)
   at_predicted=at_outputs.cpu()
   attention=at_predicted.detach()
   
   img=util.imshape(images[0,0,:,:,:])

   #attention map
   heatmap = attention[0,:,:,:]
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
   image_list=[img,heatmap,s_img]
   fig = plt.figure(figsize=(10, 10))
   for i,data in enumerate(image_list):
      fig.add_subplot(1, 3, i+1)
      plt.imshow(data)
      
   plt.savefig(os.path.join("result","image",save_name+".png"))
   #plt.show()
   
 