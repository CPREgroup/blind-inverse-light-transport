import torch
import torch.nn as nn
from tools import device
from positionalencoding import *
from functools import partial
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # torch_init.xavier_uniform_(m.weight)
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)
class Upsample2(nn.Module):
    def __init__(self, mode='nearest'):
        super(Upsample2, self).__init__()
        self.mode = mode

    def forward(self, x):
        xs = x.shape
        # x = nn.functional.interpolate(x, scale_factor=2, align_corners=True, mode=self.mode)
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.mode)
        return x
class TNet_SV(nn.Module):

    def __init__(self, svecs, Zmean, out_channels=1, neural=True):
        super(TNet_SV, self).__init__()

        self.neural = neural
        self.out_channels = out_channels
        self.S0 = [32,2,2]  # 32 feature channels in 2x2 seed "image"
        #
        # # the input to the network (trainable, so let's just start with zeros (should maybe be randn but probably not broken))
        self.x0 = torch.nn.Parameter(torch.zeros([1] + self.S0, requires_grad=True,device=device))

        self.svs = svecs.permute(0,2,3,1)   # c, I, J, sv  (TODO, clean up unnecessary permutations etc.)
        self.s_c = self.svs.shape[0]
        self.s_I = self.svs.shape[1]
        self.s_J = self.svs.shape[2]
        self.nvec = self.svs.shape[3]
        self.svs = self.svs.view(1,self.s_c, self.s_I*self.s_J, self.nvec)  # ready format for channel-batched matmul

        self.Zmean = Zmean.permute(2,0,1).view(1,self.s_c,self.s_I,self.s_J,1,1)

        if neural:
            layerwidth = 64
            self.svweight = torch.nn.Parameter(
                torch.tensor(np.ones([1,self.out_channels,self.svs.shape[3],1]).astype(np.float), dtype=torch.float,
                             requires_grad=True,device=device))


            # NOTE: here we are assuming a fixed number of upsamples, so this effectively sets the target resolution to 16x16!
            # If other resolutions are desired, the code needs to be changed (or ideally generalized a bit).
            self.layers = \
                nn.Sequential(
                    TConv([self.S0[0], layerwidth//2]),     # note currently coords=True by default
                    Upsample2(),
                    TConv([layerwidth//2, layerwidth]),
                    TConv([layerwidth, layerwidth]),
                    TConv([layerwidth, layerwidth]),
                    Upsample2(),
                    TConv([layerwidth, layerwidth], window=True),
                    TConv([layerwidth, layerwidth], window=True),
                    TConv([layerwidth, layerwidth], window=True),
                    Upsample2(),
                    TConv([layerwidth, layerwidth*2], coords=False),
                    TConv([layerwidth*2, layerwidth*4], coords=False),
                    TConv([layerwidth*4, self.out_channels * self.nvec], coords=False, activation=None),
                )

        else:
            # We aren't using this, but here's the optional way where we optimize directly over the entries of A instead of using the CNN.
            # It doesn't completely fail (depending on other parts), but is much less reliable.
            A0 = np.random.normal(0,1,[self.svs.shape[3], 16*16]).astype(np.float)
            A0[0, :] = 0
            A0 = A0 * 0.1

            self.A = torch.nn.Parameter(
                torch.tensor(A0, dtype=torch.float, requires_grad=True,device=device))

    def forward(self):
        if self.neural:
            # run the network
            x = self.layers(self.x0)
            xs = x.shape
            x = x.view(xs[0],self.out_channels,self.nvec,xs[2]*xs[3])     # batch, c, s, ij
            x = x * self.svweight
            self.A = x  # just store and expose for viz
            x = torch.matmul(self.svs, x)           # batch, c, I, J, ij
            xs = x.shape
            x = x.view(1,self.out_channels,self.s_I,self.s_J,16,16)  # XXX hardcoded assumption of 16x16, see also comment above at layer definitions
            # so now: [1, c, I, J, i, j]
            x = x + self.Zmean  # add the mean image by default, so the network only needs to care about generating the difference from mean (which is in span of SV's anyway)
        else:
            x = torch.matmul(self.svs, self.A)
            xs = x.shape
            x = x.view(xs[0],1,xs[1],xs[2],16,16).permute(0,1,4,5,2,3)
            x = x + self.Zmean

        return x, self.A
class TConv(nn.Module):
    def __init__(self, sizes,
                 activation=partial(nn.functional.leaky_relu, negative_slope=0.1),
                 fsize=(3, 3), auxfeats=None, coords=True, padfunc=nn.ReplicationPad2d,
                 noise=False, window=False, linear_window=False):
        super(TConv, self).__init__()
        self.activation = activation
        self.sizes = sizes
        self.fsize = fsize

        self.coords = coords
        self.coords_dim = 2 if coords else 0
        self.coords_cached = False

        self.noise = noise
        self.noise_dim = 4 if noise else 0
        self.noise_cached = False

        self.window = window
        self.linear_window = linear_window
        self.window_cached = False

        self.pad = padfunc((fsize[0] - fsize[0] // 2 - 1, fsize[0] // 2, fsize[1] - fsize[1] // 2 - 1, fsize[1] // 2))

        self.convs = nn.ModuleList()
        for s_in, s_out in zip(self.sizes, self.sizes[1:]):
            self.convs.append(nn.Conv2d(s_in + self.coords_dim + self.noise_dim, s_out, self.fsize, bias=True))

        for conv in self.convs:
            nn.init.xavier_normal(conv.weight)
            # conv.bias.data.fill_(0.000)
            nn.init.normal(conv.bias, std=0.001)

    def forward(self, x):
        xs = x.shape

        if self.coords:
            # If this is the first time this layer is evaluated, generate the linear gradient coordinate feature maps.
            if not self.coords_cached:
                self.ci = torch.linspace(-1, 1, xs[2], device=device).cuda().view(1, 1, xs[2], 1).expand(xs[0], -1, -1,
                                                                                                         xs[3])
                self.cj = torch.linspace(-1, 1, xs[3], device=device).cuda().view(1, 1, 1, xs[3]).expand(xs[0], -1,
                                                                                                         xs[2], -1)
                self.coords_cached = True
            # Thereafter just cat these cached maps onto the features
            x = torch.cat((x, self.ci, self.cj), dim=1)
            xs = x.shape

        if self.noise:
            # We could insert (fixed) random noise feature maps, but disabled in current version
            if not self.noise_cached:
                self.N = torch.randn((1, self.noise_dim, xs[2], xs[3]), device=device).cuda().expand(
                    (xs[0], -1, -1, -1))
                self.noise_cached = True
            x = torch.cat((x, self.N), dim=1)
            xs = x.shape

        for conv in self.convs:
            x = self.pad(x)
            x = conv(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.window:
            if not self.window_cached:
                # Similar caching mechanism as with coords above.
                # The linear_window version is not used.
                if not self.linear_window:
                    W = np.matmul(
                        np.reshape(np.hanning(xs[2]), [xs[2], 1]),
                        np.reshape(np.hanning(xs[3]), [1, xs[3]]))
                    W = np.reshape(W, [1, 1, xs[2], xs[3]])
                    W = np.sqrt(W)
                    self.W = torch.from_numpy(W).float().to(device)
                else:
                    Wi = torch.linspace(0, 2, xs[2], device=device).cuda().view(1, 1, xs[2], 1).expand(xs[0],
                                                                                                       xs[1] // 4, -1,
                                                                                                       xs[3])
                    Wj = torch.linspace(0, 2, xs[3], device=device).cuda().view(1, 1, 1, xs[3]).expand(xs[0],
                                                                                                       xs[1] // 4,
                                                                                                       xs[2], -1)
                    self.W = torch.cat((Wi, Wj, 1 - Wi, 1 - Wj), dim=1)

                self.window_cached = True

            x = x * self.W

        return x
