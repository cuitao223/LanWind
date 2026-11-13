import torch
from torch import nn

class ConvLSTMCell(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__();self.hidden=hidden
        self.gates=nn.Conv2d(inputs+hidden,4*hidden,3,padding=1)
    def forward(self,x,state=None):
        if state is None:
            h=x.new_zeros(x.shape[0],self.hidden,*x.shape[-2:]);c=torch.zeros_like(h)
        else:h,c=state
        i,f,o,g=self.gates(torch.cat([x,h],1)).chunk(4,1)
        c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        h=torch.sigmoid(o)*torch.tanh(c)
        return h,c

