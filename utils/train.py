import torch
import os
from tensorboardX import SummaryWriter


def model_train(trainloader,net,criterion,optimizer,device,num_epoch,file_name,two_stream=False):
   #tensorboard file_path
   writer = SummaryWriter(os.path.join("runs",file_name))
   #training
   for epoch in range(num_epoch):  
  
       global_i=epoch*len(trainloader)
       running_loss = 0.0
    
       for i, data in enumerate(trainloader, 0):
        
           global_i+=1
        
           # get the inputs
           images, targets, labels, poses = data
           images, labels = images.to(device), labels.to(device)
           if two_stream is not False:
               poses = poses.to(device)
           # zero the parameter gradients
           optimizer.zero_grad()

           # forward + backward + optimize
           if two_stream is not False:
               outputs = net(images,poses)
           else:
               outputs = net(images)
           loss = criterion(outputs, labels.view(-1))
           loss.backward()
           optimizer.step()

           # print statistics
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
               torch.save(net.state_dict(),os.path.join("weight","main",file_name,str(epoch+1).zfill(2)+"-"+str(i+1).zfill(5)+".pth"))

   print('Finished Training')
   writer.close()
   
   #save weight 
   torch.save(net.state_dict(),os.path.join("weight","main",file_name,"finish.pth"))


def regression_train(trainloader,net,criterion,optimizer,device,num_epoch,file_name):
   #tensorboard file_path
   writer = SummaryWriter(os.path.join("runs","reg_"+file_name))
   #training
   for epoch in range(num_epoch):  
  
       global_i=epoch*len(trainloader)
       running_loss = 0.0
    
       for i, data in enumerate(trainloader, 0):
        
           global_i+=1
        
           # get the inputs
           images, targets, labels, poses = data
           poses=poses.view(-1,4)
           poses[:,0]=(poses[:,0]/1920)
           poses[:,1]=(poses[:,1]/1080)
           poses[:,2]=(poses[:,2]/1920)
           poses[:,3]=(poses[:,3]/1080)
           images, poses = images.to(device), poses.to(device)

           # zero the parameter gradients
           optimizer.zero_grad()

           # forward + backward + optimize
           outputs = net(images)
           loss = torch.nn.functional.mse_loss(outputs, poses)
           loss.backward()
           optimizer.step()

           # print statistics
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
               if not os.path.exists(os.path.join("weight","reg",file_name+"/")):
                   os.makedirs(os.path.join("weight","reg",file_name+"/"))
               torch.save(net.state_dict(),os.path.join("weight","reg",file_name,str(epoch+1).zfill(2)+"-"+str(i+1).zfill(5)+".pth"))

   print('Finished Training')
   writer.close()

   #save weight
   torch.save(net.state_dict(),os.path.join("weight","reg",file_name,"finish.pth"))