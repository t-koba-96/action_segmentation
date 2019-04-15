import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from utils import util
from . import cnn


#vgg with attention layer
class r_at_vgg(nn.Module):
    def __init__(self,class_num):
        super(r_at_vgg, self).__init__()
        features = list(cnn.at_vgg(class_num).features)
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

