import torch
import os
from tensorboardX import SummaryWriter


def model_train(trainloader,net,criterion,optimizer,device,num_epochs,file_name,two_stream=False):
   #tensorboard file_path
   writer = SummaryWriter(os.path.join("runs",file_name))
   #training
   for epoch in range(num_epochs):  
  
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

   print('Finished Training')
   writer.close()

   #save weight
   torch.save(net.state_dict(),os.path.join("weight",file_name+".pth"))


def regression_train(trainloader,net,criterion,optimizer,device,num_epochs,file_name):
   #tensorboard file_path
   writer = SummaryWriter(os.path.join("runs",file_name))
   #training
   for epoch in range(num_epochs):  
  
       global_i=epoch*len(trainloader)
       running_loss = 0.0
    
       for i, data in enumerate(trainloader, 0):
        
           global_i+=1
        
           # get the inputs
           images, targets, labels, poses = data
           poses=poses.view(-1,4)
           poses[:,0]=(poses[:,0]/1920-0.5)
           poses[:,1]=(poses[:,1]/1080-0.5)
           poses[:,2]=(poses[:,2]/1920-0.5)
           poses[:,3]=(poses[:,3]/1080-0.5)
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

   print('Finished Training')
   writer.close()

   #save weight
   torch.save(net.state_dict(),os.path.join("weight",file_name+".pth"))