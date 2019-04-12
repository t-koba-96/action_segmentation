import torch
import torch.nn as nn
from torchvision.models import vgg16
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

#attention_vgg+lstm
class vgg_lstm(nn.Module):
    def __init__(self,class_num):
        super(vgg_lstm, self).__init__()
        features = list(cnn.at_vgg(class_num).features)
        self.features = nn.ModuleList(features)

        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))

        self.rnn_classify = rnn.lstm_layer(class_num)
        
        
    def forward(self, x):
        
        batch_size=x.size(0)
        clip_length=x.size(1)
        
        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
            x = model(x)

        x = self.avgpool(x)
            
        x = x.view(x.size(0), -1)
            
        x = x.view(batch_size,clip_length,-1)

        x = self.rnn_classify(x)
            
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

        self.img_tcn = tcn.Dilated_TCN(512,[256,128,64,32,class_num])

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
        
        x = self.img_tcn(x)

        x = x.permute(0,2,1).view(-1,x.size(1))
        
        y = self.pose_tcn(y.permute(0,2,1))
        
        y = y.permute(0,2,1).view(-1,y.size(1))

        x = torch.add(x,y)

        return x



# dual_attention_tcn 修正ひつよう
class dual_attention_tcn(nn.Module):
    def __init__(self,class_num,net):
        super(dual_attention_tcn, self).__init__()
        pose_at=[]
        for i in range(27):
           pose_at.append((net.module.features)[i])
        self.pose_at = nn.ModuleList(pose_at)

        pose_feat=[]
        for i in range(3):
           pose_feat.append((net.module.features)[i+28])
        self.pose_feat = nn.ModuleList(pose_feat)

        features = list(cnn.at_vgg(class_num).features)
        self.features = nn.ModuleList(features)

        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))

        self.img_tcn = tcn.Dilated_TCN(512,[256,128,64,32,class_num])

        self.pose_tcn = tcn.Dilated_TCN(512,[256,128,64,32,class_num])

        self.attention = temporal_attention()
        
        
    def forward(self, x):
        
        batch_size=x.size(0)
        clip_length=x.size(1)
        
        x=x.view(-1,3,x.size(3),x.size(4))
        
        with torch.no_grad():
           for ii,model in enumerate(self.pose_at):
              y = model(x)
        for ii,model in enumerate(self.pose_feat):
           y = model(y)
        y = self.avgpool(y)
        y = y.view(y.size(0) , -1)
        # batch,feature,clip
        y = y.view(batch_size,clip_length,-1).permute(0,2,1)
        y = self.img_tcn(y)
        y = self.pose_tcn(y.permute(0,2,1))
        y = y.permute(0,2,1).view(-1,y.size(1))
        
        for ii,model in enumerate(self.features):
            x = model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0) , -1)
        # batch,feature,clip
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        x = self.img_tcn(x)
        x = x.permute(0,2,1).view(-1,x.size(1))

        x = torch.add(x,y)

        return x
