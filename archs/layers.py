import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange


# class SqueezeExcitation(nn.Module):
#     def __init__(self, dim, shrinkage_rate = 0.25):
#         super().__init__()
#         hidden_dim = int(dim * shrinkage_rate)
#
#         self.gate = nn.Sequential(
#             Reduce('b c h w -> b c', 'mean'),
#             nn.Linear(dim, hidden_dim, bias = False),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, dim, bias = False),
#             nn.Sigmoid(),
#             Rearrange('b c -> b c 1 1')
#         )
#
#     def forward(self, x):
#         return x * self.gate(x)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# ASAU and ACAU operation
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel,feature_dim, filter=False):
        super(ResBlock, self).__init__()

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True, bias=False)
        self.dwconv3 = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1, groups= out_channel)
        self.out = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,groups= out_channel)
        self.conv2 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=False, bias=False)

        # 是否采用空间注意力算子
        if filter:
            self.cubic_meso = cubic_attention(in_channel//2, group=1, kernel=11)
            self.cubic_global = cubic_attention_global(in_channel//2, group=1, kernel=feature_dim//2)
        self.filter = filter

        self.channel_attn_3 = depth_channel_att(in_channel, kernel=3)
        self.channel_attn_glo = depth_channel_att_glo(in_channel, kernel=3)
        
    def forward(self, x):

        x = self.conv1(x)  # local
        x = self.dwconv3(x)
        out_local = x  # 获得局部特征   416 32 32

        ########## 空间注意力 #########
        if self.filter:
            out_meso_fea,out_global_fea = torch.chunk(out_local,2,dim=1) # 208 32 32
            out_meso = self.cubic_meso(out_meso_fea)  #  208 32 32
            out_global = self.cubic_global(out_global_fea)  #  208 32 32

        out = torch.cat((out_meso,out_global),dim=1)  # 多尺度的全局特征融合  416 32 32
        out_global = self.conv2(out)

        ########## 通道注意力 #########
        out_chan_3 = self.channel_attn_3(out_global)
        out_chan = self.channel_attn_glo(out_chan_3)

        return out_chan


# ASAU and ACAU operation
class ResBlock_shallow(nn.Module):
    def __init__(self, in_channel, out_channel, feature_dim, filter=False):
        super(ResBlock_shallow, self).__init__()

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True, bias=False)
        self.dwconv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=out_channel)
        self.out = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1,groups=out_channel)
        self.conv2 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=False, bias=False)

        # 是否采用空间注意力算子
        if filter:
            self.cubic_meso7 = cubic_attention(in_channel // 2, group=1, kernel=7)
            self.cubic_meso11 = cubic_attention(in_channel // 2, group=1, kernel=11)
        self.filter = filter

        self.channel_attn_3 = depth_channel_att(in_channel, kernel=3)
        self.channel_attn_glo = depth_channel_att_glo(in_channel, kernel=3)

    def forward(self, x):

        x = self.conv1(x)  # local
        x = self.dwconv3(x)
        out_local = x  # 获得局部特征   416 32 32
        ########## 空间注意力 #########
        if self.filter:
            out_meso_fea7, out_meso_fea11 = torch.chunk(out_local, 2, dim=1)  # 208 32 32
            out_meso7 = self.cubic_meso7(out_meso_fea7)  # 208 32 32
            out_meso11 = self.cubic_meso11(out_meso_fea11)  # 208 32 32

        out = torch.cat((out_meso7, out_meso11), dim=1)  # 多尺度的全局特征融合  416 32 32
        out_global = self.conv2(out)

        ########## 通道注意力 #########
        out_chan_3 = self.channel_attn_3(out_global)
        out_chan = self.channel_attn_glo(out_chan_3)


        return out_chan


class cubic_attention(nn.Module):    # 16  1  11
    def __init__(self, dim, group, kernel):
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):  # 16 256 256
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


#
class spatial_strip_att(nn.Module):  # 16 1 11
    def __init__(self, dim, kernel=5, group=2, H=True):
        super().__init__()

        self.k = kernel  # 11
        pad = kernel // 2  # 5
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):  #  16 256 256
        filter = self.ap(x)  # 16 1 1
        filter = self.conv(filter)  #  11 1 1
        n, c, h, w = x.shape  # 16 256 256
        # 这里就是把输入x 先填充下变成 16 256 266 ，然后unfold只卷不积，变成16*11 256*256，之后在reshape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w) # n 1 16 11 256*256
        n, c1, p, q = filter.shape   # n 11 1 1
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)  # reshape( n, 1, 11, 1 )  --->  n 1 1 11 1
        filter = self.filter_act(filter)  # n 1 1 11 1
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out



class cubic_attention_global(nn.Module):    # 16  1  11
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att_global1 = spatial_strip_att_global(dim, group=group, kernel=kernel)
        self.W_spatial_att_global1 = spatial_strip_att_global(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):  # 16 256 256   || 208 32 32
        out = self.H_spatial_att_global1(x)
        out = self.W_spatial_att_global1(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att_global(nn.Module):  # 16 1 11
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel  # 11  ||  16
        pad = kernel - 1  # 5  ||  15
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)

        self.group = group # if 8
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)  # 16*8
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):  #  16 256 256 || 208 32 32
        filter = self.ap(x)  # 16 1 1  || 208 1 1
        filter = self.conv(filter)  #  11 1 1     16 1 1  ||   16 1 1  16*8 1 1
        n, c, h, w = x.shape  # 16 256 256  || 208 32 32
        # 这里就是把输入x 先填充下变成 16 256 266 ，然后unfold只卷不积，变成16*11 256*256，之后在reshape
        x = F.unfold(self.pad(x), dilation=2,stride=1, kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w) # n 1 16 11 256*256   || n 1 208 16 32*32  n 8 26 16 32*32
        n, c1, p, q = filter.shape   # n 11 1 1  || 16 1 1
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)  # reshape( n, 1, 11, 1 )  --->  n 1 1 11 1   ||   n 1 1 16 1   n 8 1 16 1
        filter = self.filter_act(filter)  # n 1 1 11 1 ||  n 1 1 16 1     n 8 1 16 1
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class depth_channel_att(nn.Module):
    def __init__(self, dim, kernel=3) -> None:

        # if dim == 52 or 104:
        #     self.dim = dim // 13
        # else:
        self.dim = dim

        super().__init__()
        self.kernel = (1, kernel)  # ( 1 3 )
        pad_r = pad_l = kernel // 2  # 1
        self.pad = nn.ReflectionPad2d((pad_r, pad_l, 0, 0))
        self.conv = nn.Conv2d(self.dim, kernel * self.dim, kernel_size=1, stride=1, bias=False, groups=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.filter_act = nn.Sigmoid()  # 注意这里使用的是tanh
        self.filter_bn = nn.BatchNorm2d(kernel* self.dim)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):  # 32 256 256           416 32 32   1248 1 1
        # x_orig = x
        # x = self.dwconv(x)  # 26 32 32  # 13 64 64  #

        filter = self.filter_bn(self.conv(self.gap(x)) ) # 96 1 1      1248 1 1   || 26*3=78 1 1
        filter = self.filter_act(filter)  # 96 1 1    1248 1 1  ||
        b, c, h, w = filter.shape  # 3 96 1 1
        filter = filter.view(b, self.kernel[1], c//self.kernel[1], h*w).contiguous().permute(0, 1, 3, 2).contiguous()  # 3 3 32 1 --> 3 3 1 32  || 3 3 1 26
        B, C, H, W = x.shape  # 3 32 256 256  || 3 26 32 32
        out = x.permute(0, 2, 3, 1).view(B, H*W, C).contiguous().unsqueeze(1)  # 3 256 256 32 --> 3 256*256 32--> 3 1 256*256 32  || 3 1 32*32 26
        out = F.unfold(self.pad(out), kernel_size=self.kernel, stride=1)   # 3 3 2097152(256*256*32) || 3 3
        out = out.view(B, self.kernel[1], H*W, -1).contiguous()  # 3 3 256*256 32  || 3 3 32*32 26
        out = torch.sum(out * filter, dim=1, keepdim=True).permute(0,3,1,2).contiguous().reshape(B,C,H,W)  # 3 1 256*256 32---> 3 32 1 256*256--> 3 32 256 256 || 3 1 32*32 26 -> 3 26 32 32

        # out = self.proj(out) # 3 416 32 32

        out = out * self.gamma + x * self.beta
        return out    # 自适应的残差连接


class depth_channel_att_glo(nn.Module):
    def __init__(self, dim, kernel=3) -> None:

        if dim == 52 or 104:
            self.dim = dim // 13
        else:
            self.dim = dim // 16

        super().__init__()
        self.kernel = (1, kernel)  # ( 1 3 )
        pad_r = pad_l = kernel // 2  # 1
        self.pad = nn.ReflectionPad2d((pad_r, pad_l, 0, 0))
        self.conv = nn.Conv2d(self.dim, kernel * self.dim, kernel_size=1, stride=1, bias=False, groups=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.filter_act = nn.Sigmoid()  # 注意这里使用的是tanh
        self.filter_bn = nn.BatchNorm2d(kernel* self.dim)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))
        if dim == 52 or 104:
            self.dwconv = nn.Conv2d(dim,dim//13,kernel_size=1,stride=1,padding=0,groups=dim//13,bias=False )  # dim//16 = 416//16=26
            self.proj = nn.Conv2d(dim//13,dim,kernel_size=1,stride=1,padding= 0,bias=False)
        else:
            self.dwconv = nn.Conv2d(dim, dim // 16, kernel_size=1, stride=1, padding=0, groups=dim // 16,bias=False)  # dim//16 = 416//16=26
            self.proj = nn.Conv2d(dim // 16, dim, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):  # 32 256 256           416 32 32   1248 1 1
        x_orig = x
        x = self.dwconv(x)  # 26 32 32  # 13 64 64  #

        filter = self.filter_bn(self.conv(self.gap(x)) ) # 96 1 1      1248 1 1   || 26*3=78 1 1
        filter = self.filter_act(filter)  # 96 1 1    1248 1 1  ||
        b, c, h, w = filter.shape  # 3 96 1 1
        filter = filter.view(b, self.kernel[1], c//self.kernel[1], h*w).contiguous().permute(0, 1, 3, 2).contiguous()  # 3 3 32 1 --> 3 3 1 32  || 3 3 1 26
        B, C, H, W = x.shape  # 3 32 256 256  || 3 26 32 32
        out = x.permute(0, 2, 3, 1).view(B, H*W, C).contiguous().unsqueeze(1)  # 3 256 256 32 --> 3 256*256 32--> 3 1 256*256 32  || 3 1 32*32 26
        out = F.unfold(self.pad(out), kernel_size=self.kernel, stride=1)   # 3 3 2097152(256*256*32) || 3 3
        out = out.view(B, self.kernel[1], H*W, -1).contiguous()  # 3 3 256*256 32  || 3 3 32*32 26
        out = torch.sum(out * filter, dim=1, keepdim=True).permute(0,3,1,2).contiguous().reshape(B,C,H,W)  # 3 1 256*256 32---> 3 32 1 256*256--> 3 32 256 256 || 3 1 32*32 26 -> 3 26 32 32

        out = self.proj(out) # 3 416 32 32

        out = out * self.gamma + x_orig * self.beta
        return out    # 自适应的残差连接
