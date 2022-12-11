import torch
from torch import nn
import torch.nn.functional as F
withposi = True
separate = False
from tools import device
xydim = 6

class ResBlock(torch.nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        num_filter = 32
        # self.numfilter = num_filter
        self.conv1 = torch.nn.Conv3d(6, num_filter, (3, 3,3), padding=1)
        self.conv2 = torch.nn.Conv3d(num_filter,num_filter * 2,(3, 3, 3), padding=1)
        self.conv3 = torch.nn.Conv3d(num_filter * 2,num_filter * 2, (3, 3, 3), padding=1)
        self.conv4 = torch.nn.Conv3d(num_filter * 2, num_filter, (3, 3,3), padding=1)
        self.conv5 = torch.nn.Conv3d(num_filter, 3, (3, 3, 3), padding=1)

    def forward(self, x_input):
        y = x_input
        y = F.leaky_relu(self.conv1(y))
        y = F.leaky_relu(self.conv2(y))
        y = F.leaky_relu(self.conv3(y))
        y = F.leaky_relu(self.conv4(y))
        y = self.conv5(y)

        result = x_input[:, 0:3, :, :] + y

        return result

class BasicStage(torch.nn.Module):
    def __init__(self, index, args):
        super(BasicStage, self).__init__()

        global xydim, withposi
        if args.mis == 'allpe':
            xydim = args.L * 4 + 2
        elif args.mis == 'xyonly':
            xydim = 2
        elif args.mis == 'nope':
            withposi = False


        self.args = args
        self.index = index

        A = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(A, 1e-1)
        self.lamda = nn.Parameter(A,requires_grad=True).to(device)

        B = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(B, 1e-7)
        self.alpha = nn.Parameter(B,requires_grad=True).to(device)

        self.denoisenet1 = ResBlock()
        self.denoisenet2 = ResBlock()

    def forward(self,u,v,z,tv,tT,pos,s_t):

        v_next = v - F.relu(self.alpha) * (
                torch.matmul(tT, tv - z).view(1, 3, 16, 16, s_t) + self.lamda * (
                    v - u)
        )

        if withposi:
            # u_next = self.denoisenet1(v_next)
            # u_next = self.denoisenet2(u_next)
            u_next = self.denoisenet1(torch.cat((v_next, pos),dim=1))
            u_next = self.denoisenet2(torch.cat((u_next, pos),dim=1))
        else:
            u_next = self.denoisenet1(v_next)
            u_next = self.denoisenet2(u_next)

        return v_next, u_next

class VRcnn(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        super(VRcnn, self).__init__()
        onelayer = []
        self.LayerNo = self.args.layer_num

        for i in range(self.LayerNo):
            onelayer.append(BasicStage(i, args))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self,u,v,z,t1, tT,pos,s_t):

        for i in range(self.LayerNo):
            v1 = v.view(1, 3, 16 * 16, s_t)
            tv = torch.matmul(t1,v1)

            v, u = self.fcs[i](u,v,z,tv,tT,pos,s_t)
        return u