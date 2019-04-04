import torch
import torch.nn as nn
from torchvision.models import vgg16


#lstm layer
class lstm_layer(nn.Module):
    def __init__(self,class_num):
        super(lstm_layer, self).__init__()
        
        #(input_size,hidden_size)
        self.rnn = nn.LSTM(512,256,batch_first = True)
        
        #(hidden_size,output_classes)
        self.output_layer = nn.Linear(256,11)

        self.class_num = class_num 

    def forward(self, x, h0=None):
        self.rnn.flatten_parameters()
        x , h = self.rnn(x,h0) 
        x = self.output_layer(x) 

        x=x.view(-1,self.class_num)

        return x

