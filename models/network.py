import torch
import torch.nn as nn
from torchvision.models import vgg16,resnet50
import torch.nn.functional as F
from . import cnn,rnn,tcn

# if using crossentropyloss  , output size (batch*sequence , class_num)

# temporal attention  
class temporal_attention(nn.Module):
    def __init__(self):
        super(temporal_attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):    
        y = self.softmax(torch.matmul(x,x.permute(0,2,1)))
        x = (torch.matmul(y,x)).permute(0,2,1)

        return x


# resnet + dilated tcn
class vgg_tcn(nn.Module):
    def __init__(self,class_num):
        super(resnet_tcn, self).__init__()
        features = list(vgg16(pretrained = True).features)
        self.features = nn.ModuleList(features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.tcn = tcn.Dilated_TCN(512,[256,128,64,32,class_num])
        self.attention = temporal_attention()
        
    def forward(self, x):
        batch_size=x.size(0)
        clip_length=x.size(1)
        x=x.view(-1,3,x.size(3),x.size(4))
        for ii,model in enumerate(self.features):
            x = model(x)
        x= self.avgpool(x)
        x = x.view(x.size(0) , -1)    
        # batch,clip,feature
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        x = self.tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))
            
        return x



# resnet + dilated tcn
class resnet_tcn(nn.Module):
    def __init__(self,class_num):
        super(resnet_tcn, self).__init__()
        resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.maxpool = nn.MaxPool2d(kernel_size=7)
        self.tcn = tcn.Dilated_TCN(2048,[512,256,128,64,class_num])
        self.attention = temporal_attention()
        
    def forward(self, x):
        batch_size=x.size(0)
        clip_length=x.size(1)
        x=x.view(-1,3,x.size(3),x.size(4))
        x=self.resnet(x)
        x = self.maxpool(x)       
        x = x.view(x.size(0) , -1)    
        # batch,clip,feature
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        x = self.tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))
            
        return x


# attention_vgg + dilated tcn
class attention_tcn(nn.Module):
    def __init__(self,class_num):
        super(attention_tcn, self).__init__()
        features = list(cnn.at_vgg(class_num).features)
        self.features = nn.ModuleList(features)
        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))
        self.tcn = tcn.Dilated_TCN(512,[256,128,64,32,class_num])
        self.attention = temporal_attention()
              
    def forward(self, x):    
        batch_size=x.size(0)
        clip_length=x.size(1)  
        x=x.view(-1,3,x.size(3),x.size(4)) 
        for ii,model in enumerate(self.features):
            x = model(x)
        x = self.avgpool(x)           
        x = x.view(x.size(0) , -1)          
        # batch,clip,feature
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)     
        x = self.tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))
            
        return x



# two-stream tcn
class twostream_tcn(nn.Module):
    def __init__(self,class_num):
        super(twostream_tcn, self).__init__()
        features = list(cnn.at_vgg(class_num).features)
        self.features = nn.ModuleList(features)
        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))
        self.img_tcn = tcn.Dilated_TCN(516,[256,128,64,32,class_num])
        self.pose_tcn = tcn.Dilated_TCN(4,[256,128,64,32,class_num])
        self.attention = temporal_attention()
        
        
    def forward(self, x, y):
        batch_size=x.size(0)
        clip_length=x.size(1)
        x=x.view(-1,3,x.size(3),x.size(4))
        for ii,model in enumerate(self.features):
            x = model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0) , -1)
        # batch,feature,clip
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        x=torch.cat([x,y.permute(0,2,1)], dim=1)
        x = self.img_tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))
        #y = self.pose_tcn(y.permute(0,2,1))
        #y = y.permute(0,2,1).view(-1,y.size(1))
        #x = torch.add(x,y)

        return x



# cropping hand 
class cutout_tcn(nn.Module):
    def __init__(self,class_num):
        super(cutout_tcn, self).__init__()
        features = list(cnn.at_vgg(class_num).features)
        self.features_1 = nn.ModuleList(features)
        self.features_2 = nn.ModuleList(features)
        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))
        self.tcn = tcn.Dilated_TCN(1024,[256,128,64,32,class_num])
        self.attention = temporal_attention()
        
    def forward(self, x , y):
        batch_size=x.size(0)
        clip_length=x.size(1)
        x=x.view(-1,3,x.size(3),x.size(4))
        for ii,model in enumerate(self.features_1):
             x = model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0) , -1)
        y = y.view(-1,3,y.size(3),y.size(4))
        for ii,model in enumerate(self.features_2):
             y = model(y)
        y = self.avgpool(y)
        y = y.view(y.size(0) , -1)
        x = torch.cat([x,y] , dim=1)
        # batch,clip,feature
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        x = self.tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))
            
        return x


# posemap as attention
class posemap_tcn(nn.Module):
    def __init__(self,class_num):
        super(posemap_tcn, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features)
        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))
        self.tcn = tcn.Dilated_TCN(512,[256,128,64,32,class_num])
        
    def forward(self, x, y):
        batch_size=x.size(0)
        clip_length=x.size(1)
        x=x.view(-1,3,x.size(3),x.size(4))
        for ii,model in enumerate(self.features):
            if ii == 5:
                 x = x*y+x
            x = model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0) , -1)
        # batch,clip,feature
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        x = self.tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))
            
        return x