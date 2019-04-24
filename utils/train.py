import torch
import os
import cv2
from tensorboardX import SummaryWriter
from . import util



def model_train(trainloader,testloader,net,criterion,optimizer,device,num_epoch,file_name,cutout_img=False,two_stream=False,posemap=False):
   #tensorboard file_path
   writer = SummaryWriter(os.path.join("runs",file_name))
   #training
   for epoch in range(num_epoch):  
  
       global_i=epoch*len(trainloader)
       running_loss = 0.0
    
       for i, data in enumerate(trainloader, 0):
        
           global_i+=1
        
           if cutout_img is not False:
                 images, left_img, right_img, targets, labels, poses = data
                 left_img, right_img, labels = left_img.to(device), right_img.to(device), labels.to(device)
           else:
                 images, targets, labels, poses = data
                 images, labels = images.to(device), labels.to(device)
           if two_stream is not False:
                 if posemap is not False:
                     poses=poses.view(-1,4)
                     posemap=[]
                     for x in range(poses.size(0)):
                         map=torch.from_numpy(util.pose_map(poses[x,:],224,31))
                         posemap.append(map)
                     posemap = torch.stack(posemap, 0).view(-1,1,224,224)
                     posemap = posemap.to(device)
                 else:
                     poses = poses.to(device)
           if posemap is not False:
                 if two_stream is False:
                     poses=poses.view(-1,4)
                     posemap=[]
                     for x in range(poses.size(0)):
                         map=torch.from_numpy(cv2.resize(util.pose_map(poses[x,:],224,251),(112,112)))
                         posemap.append(map)
                     posemap = torch.stack(posemap, 0).view(-1,1,112,112)
                     posemap = posemap.to(device)


           optimizer.zero_grad()

           if cutout_img is not False:
                 if two_stream is not False:
                     outputs = net(left_img,right_img,poses)
                 else:
                     outputs = net(left_img,right_img)
           else:
                 if two_stream is not False:
                     if posemap is False:
                         outputs = net(images,poses)
                 elif posemap is not False:
                     outputs = net(images,posemap)
                 else:
                     outputs = net(images)
                     
           loss = criterion(outputs, labels.view(-1))
           loss.backward()
           optimizer.step()

           # test loss 
           if i % 500 == 0:
                 test_loss = 0
                 total = 0
                 for ii, data in enumerate(testloader, 0):
                     total += 1
                     if cutout_img is not False:
                         images, left_img, right_img, targets, labels, poses = data
                         left_img, right_img, labels = left_img.to(device), right_img.to(device), labels.to(device)
                     else:
                         images, targets, labels, poses = data
                         images, labels = images.to(device), labels.to(device)
                     if two_stream is not False:
                         if posemap is not False:
                             poses=poses.view(-1,4)
                             posemap=[]
                             for x in range(poses.size(0)):
                                 map=torch.from_numpy(util.pose_map(poses[x,:],224,31))
                                 posemap.append(map)
                             posemap = torch.stack(posemap, 0).view(-1,1,224,224)
                             posemap = posemap.to(device)
                         else:
                             poses = poses.to(device)
                     if posemap is not False:
                         if two_stream is False:
                             poses=poses.view(-1,4)
                             posemap=[]
                             for x in range(poses.size(0)):
                                 map=torch.from_numpy(cv2.resize(util.pose_map(poses[x,:],224,251),(112,112)))
                                 posemap.append(map)
                             posemap = torch.stack(posemap, 0).view(-1,1,112,112)
                             posemap = posemap.to(device)
                     if cutout_img is not False:
                         if two_stream is not False:
                             outputs = net(left_img,right_img,poses)
                         else:
                             outputs = net(left_img,right_img)
                     else:
                         if two_stream is not False:
                             if posemap is False:
                                 outputs = net(images,poses)
                         elif posemap is not False:
                             outputs = net(images,posemap)
                         else:
                             outputs = net(images)
                     
                     test_loss += criterion(outputs, labels.view(-1)).item()
                
                 writer.add_scalar('test/test_loss', test_loss/total , global_i)
                
           # terminal
           running_loss += loss.item()
           if i % 200 == 199:    
                 print('[%d, %5d] loss: %.3f' %
                     (epoch + 1, i + 1, running_loss / 200))
                 running_loss = 0.0
        
           #tensorboard   (title,y,x)
           if i % 10 == 0:
                 writer.add_scalar('train/train_loss', loss.item() , global_i)

           #save weight
           if i % 200 == 0:
                 if not os.path.exists(os.path.join("weight","main",file_name+"/")):
                     os.makedirs(os.path.join("weight","main",file_name+"/"))
                 torch.save(net.state_dict(),os.path.join("weight","main",file_name,str(epoch+1).zfill(2)+"-"+str(i).zfill(5)+".pth"))


   print('Finished Training')
   writer.close()
   
   #save weight 
   torch.save(net.state_dict(),os.path.join("weight","main",file_name,"finish.pth"))
