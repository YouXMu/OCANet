import archs.common as common
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from .layers import *

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv1 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out1 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim, bias=False),
        )
    def forward(self, x,k_v):
        b,c,h,w = x.shape
        # k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v=self.kernel(k_v).view(-1,c,1,1)
        # k_v1,k_v2=k_v.chunk(2, dim=1)
        # x = x*k_v1+k_v2
        x = self.project_in1(x)
        x1, x2 = self.dwconv1(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out1(x)
        x = x+k_v
        return x

class Omni_Attention(nn.Module):

    def __init__(self,dim,feature_dim,bias):
        super(Omni_Attention, self).__init__()

        self.stp = ResBlock(dim,dim,feature_dim,filter=True)
        self.kernel = nn.Sequential(
            nn.Linear(256,dim,bias=bias)
        )

    def forward(self,x,k_v):
        b,c,h,w = x.shape
        k_v = self.kernel(k_v).view(-1,c,1,1)

        out = self.stp(x)
        out = out + k_v
        return out

class Omni_Attention_shallow(nn.Module):

    def __init__(self,dim,feature_dim,bias):
        super(Omni_Attention_shallow, self).__init__()

        self.stp1 = ResBlock_shallow(dim,dim,feature_dim,filter=True)
        self.kernel = nn.Sequential(
            nn.Linear(256,dim,bias=bias)
        )

    def forward(self,x,k_v):
        b,c,h,w = x.shape
        k_v = self.kernel(k_v).view(-1,c,1,1)

        out = self.stp1(x)
        out = out + k_v
        return out


class TransformerBlock1(nn.Module):
    def __init__(self, dim,ffn_expansion_factor,feature_dim, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Omni_Attention_shallow(dim,feature_dim, bias)

        self.norm2 = LayerNorm(dim,LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v=y[1]
        x = x + self.attn1(self.norm1(x),k_v)
        x = x + self.ffn2(self.norm2(x),k_v)

        return [x,k_v]


class TransformerBlock(nn.Module):
    def __init__(self, dim,ffn_expansion_factor,feature_dim, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Omni_Attention(dim,feature_dim, bias)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        # self.norm3 = LayerNorm(dim, LayerNorm_type)
        # self.attn2 = ChannelAttention(dim,bias)
        self.norm2 = LayerNorm(dim,LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v=y[1]
        x = x + self.attn1(self.norm1(x),k_v)
        x = x + self.ffn2(self.norm2(x),k_v)

        return [x,k_v]


class GatedEmb(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(GatedEmb, self).__init__()

        self.gate_proj1 = nn.Conv2d(in_c, embed_dim*2, kernel_size=3,stride=1,padding=1,bias=bias)


    def forward(self, x):
        x = self.gate_proj1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        # x = x1 * torch.sigmoid(gate)

        return x


class Downsample(nn.Module):  # 使用DWT()进行下采样
    def __init__(self,n_feat):
        super(Downsample, self).__init__()
        self.n_feat = n_feat
        self.body= common.DWT()
        self.out=nn.Conv2d(n_feat*4,n_feat*2,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,x):
        # x1,x2 = x.chunk(2,dim=1)
        x = self.body(x)
        return  self.out(x)


class Upsample(nn.Module):  # 使用IWT()进行上采样
    def __init__(self,n_feat):
        super(Upsample, self).__init__()

        self.body = common.IWT()
        self.out = nn.Conv2d(n_feat//4,n_feat//2,kernel_size=3,stride=1,padding=1,bias=False)  # 这里也可以用1×1的卷积来改变通道数

    def forward(self,x):
        x = self.body(x)  # 104 64 64
        return self.out(x)  # 208 64 64


class DIformer(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8,1,1,1],
        num_refinement_blocks = 4,
        ffn_expansion_factor = 2.66,
        bias = False,
        features=[256, 128, 64, 32],
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(DIformer, self).__init__()

        self.patch_embed = GatedEmb(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock1(dim=dim,  ffn_expansion_factor=ffn_expansion_factor,feature_dim = features[0], bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1),  ffn_expansion_factor=ffn_expansion_factor, feature_dim = features[1],bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2),  ffn_expansion_factor=ffn_expansion_factor,feature_dim = features[2], bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor, bias=bias, feature_dim = features[3],LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2),  ffn_expansion_factor=ffn_expansion_factor,feature_dim = features[2], bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[4])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor,feature_dim = features[1], bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[5])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock1(dim=int(dim*2**1),  ffn_expansion_factor=ffn_expansion_factor,feature_dim = features[0], bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[6])])
        
        #self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act_last = nn.Tanh()

    def forward(self, inp_img, k_v):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1,_ = self.encoder_level1([inp_enc_level1,k_v])

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _ = self.encoder_level2([inp_enc_level2, k_v])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _ = self.encoder_level3([inp_enc_level3, k_v])

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, _ = self.latent([inp_enc_level4, k_v])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3([inp_dec_level3, k_v])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2([inp_dec_level2, k_v])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,_ = self.decoder_level1([inp_dec_level1,k_v])

        # out_dec_level1,_ = self.refinement([out_dec_level1,k_v])

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1)  # + inp_img
        out_dec_level1 = self.act_last(out_dec_level1)
        out_dec_level1 = (out_dec_level1 + 1) / 2

        return out_dec_level1

class CPEN(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6):
        super(CPEN, self).__init__()
        E1=[nn.Conv2d(64, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
    def forward(self, x):
        x = self.pixel_unshuffle(x)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea2 = self.mlp(fea)
        
        return fea2

class OCANetS2(nn.Module):
    def __init__(self,         
        n_encoder_res=6,         
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8,1,1,1],
        num_refinement_blocks = 4,
        ffn_expansion_factor = 2.66,
        features=[256, 128, 64, 32],
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6,
       ):
        super(OCANetS2, self).__init__()

        # Generator
        self.G = DIformer(
        inp_channels=inp_channels, 
        out_channels=out_channels, 
        dim = dim,
        num_blocks = num_blocks, 
        num_refinement_blocks = num_refinement_blocks,
        ffn_expansion_factor = ffn_expansion_factor,
        features = features,
        bias = bias,
        LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
        dual_pixel_task = dual_pixel_task       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
)
        self.condition = CPEN(n_feats=64, n_encoder_res=n_encoder_res)

    def forward(self, img):  # 一个是masked image   一个由S1中cpen网络带来的真token
        if self.training:
            # IPR, pred_deg_list=self.diffusion(img,deg_prepT)
            IPR = self.condition(img)

            sr = self.G(img, IPR)

            # return sr, pred_deg_list
            return sr, IPR
        else:
            # IPR=self.diffusion(img)
            IPR = self.condition(img)

            sr = self.G(img, IPR)

            return sr, None
