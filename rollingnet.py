import torch
from torch import nn

from iternet import *
from positionalencoding import *

class Fullcnn(torch.nn.Module):
    def __init__(self,z,pos,s_t,args):
        super(Fullcnn, self).__init__()
        self.s_t = s_t
        self.z = z.permute(3, 1, 2, 0).view(3, 96 * 128, self.s_t).unsqueeze(0)
        self.pos = nn.Parameter(pos, requires_grad=True).to(device)
        self.args = args
        self.specsrcnn = VRcnn(args)

    def forward(self,t1,tT):
        vm = torch.matmul(tT , self.z)
        tv = torch.matmul(t1,vm)
        r = torch.sum(torch.abs(tv)) / torch.sum(torch.abs(self.z))

        v = vm / torch.pow(r,0.5)
        v = v.view(1, 3, 16, 16, self.s_t)
        u = v

        spec_cnn = self.specsrcnn(u,v,self.z, t1, tT,self.pos,self.s_t)

        return spec_cnn

