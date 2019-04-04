import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from . import cnn,rnn,tcn

# if using crossentropyloss  , output size (batch*sequence , class_num)



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

        self.tcn = tcn.Dilated_TCN(512,[64,96,class_num])
        
        
    def forward(self, x):
        
        batch_size=x.size(0)
        clip_length=x.size(1)
        
        x=x.view(-1,3,x.size(3),x.size(4))
        
        for ii,model in enumerate(self.features):
            x = model(x)

        x = self.avgpool(x)
            
        x = x.view(x.size(0) , -1)
            
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)

        x = self.tcn(x)

        x = x.permute(0,2,1).view(-1,x.size(1))
            
        return x


