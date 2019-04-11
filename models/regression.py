import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from utils import util



class r_cnn(nn.Module):
    def __init__(self):
        super(r_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256,512,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
               
        x=x.view(-1,3,x.size(3),x.size(4))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
            
        x= self.avgpool(x)

        x= x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
            
        return x




#vgg16
class r_vgg(nn.Module):
    def __init__(self):
        super(r_vgg, self).__init__()
        features = list(vgg16(pretrained = True).features)
        self.features = nn.ModuleList(features)
        
        #self.batchnorm = nn.BatchNorm2d(channel)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.classifier=nn.Sequential(
            nn.Linear(512,1024),
            nn.Tanh(),
            nn.Linear(1024,4)
            # nn.Softmax(dim=1)
        )
        
    def forward(self, x):
               
        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
            x = model(x)
            
        x= self.avgpool(x)

        x= x.view(x.size(0), -1)
        
        x= self.classifier(x)

        #x= torch(x)
            
        return x




# attention layer (sigmoid)
class r_attention_sigmoid(nn.Module):
    def __init__(self):
        super(r_attention_sigmoid, self).__init__()
        
        self.conv1 = nn.Conv2d(512,512,3,1,1)
        self.conv2 = nn.Conv2d(512,512,3,1,1)
        self.normalize = nn.InstanceNorm2d(512)

    def forward(self, x):
               
        y = torch.sigmoid(self.normalize(self.conv2(x)))

        x = F.relu((self.conv1(x))*y)

        return x




#vgg with attention layer
class r_at_vgg(nn.Module):
    def __init__(self):
        super(r_at_vgg, self).__init__()
        features=[]
        for i in range(26):
           features.append((vgg16(pretrained = True).features)[i])
        
        #attention_ReLu() or attention_sigmoid()
        features.append(r_attention_sigmoid())
        
        for i in range(3):
           features.append((vgg16(pretrained = True).features)[i+28])
        
        self.features = nn.ModuleList(features)
   
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.classifier=nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,4)
            #nn.Softmax(dim=1)
        )
    

    def forward(self, x):

        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
            x = model(x)
            
        x= self.avgpool(x)

        x= x.view(x.size(0), -1)
        
        x= self.classifier(x)

        x=torch.sigmoid(x)
            
        return x



# attention network
class r_attention_net(nn.Module):
    def __init__(self,net):
        super(r_attention_net, self).__init__()
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