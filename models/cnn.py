import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from utils import util

#vgg16
class vgg(nn.Module):
    def __init__(self,class_num):
        super(vgg, self).__init__()
        features = list(vgg16(pretrained = True).features)
        self.features = nn.ModuleList(features)
        
        #self.batchnorm = nn.BatchNorm2d(channel)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.classifier=nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,class_num)
            # nn.Softmax(dim=1)
        )
        
    def forward(self, x):
               
        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
            x = model(x)
            
        x= self.avgpool(x)

        x = x.view(x.size(0), -1)
        
        x= self.classifier(x)
            
        return x




# attention layer (sigmoid)
class attention_sigmoid(nn.Module):
    def __init__(self):
        super(attention_sigmoid, self).__init__()
        
        self.conv1 = nn.Conv2d(512,512,3,1,1)
        self.conv2 = nn.Conv2d(512,512,3,1,1)
        self.normalize = nn.InstanceNorm2d(512)

    def forward(self, x):
               
        y = torch.sigmoid(self.normalize(self.conv2(x)))

        x = F.relu((self.conv1(x))*y)

        return x




#vgg with attention layer
class at_vgg(nn.Module):
    def __init__(self,class_num):
        super(at_vgg, self).__init__()
        features=[]
        for i in range(26):
           features.append((vgg16(pretrained = True).features)[i])
        
        #attention_ReLu() or attention_sigmoid()
        features.append(attention_sigmoid())
        
        for i in range(3):
           features.append((vgg16(pretrained = True).features)[i+28])
        
        self.features = nn.ModuleList(features)
   
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.classifier=nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,class_num)
            #nn.Softmax(dim=1)
        )
    

    def forward(self, x):

        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
            x = model(x)
            
        x= self.avgpool(x)

        x = x.view(x.size(0), -1)
        
        x= self.classifier(x)
            
        return x



# attention network
class attention_net(nn.Module):
    def __init__(self,net):
        super(attention_net, self).__init__()
        features=[]
        for i in range(27):
           features.append((net.module.features)[i])

        self.features = nn.ModuleList(features)

    def forward(self, x):

        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
           
            if ii in {26}:
               #sigmoid or ReLu
               y = torch.sigmoid(model.normalize(model.conv2(x)))
               
            else:
               x = model(x)

        return y