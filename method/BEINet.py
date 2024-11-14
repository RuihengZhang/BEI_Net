import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from timm.models.layers import DropPath
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d

from utils.ops.tensor_ops import cus_sample, CARAFE

import numpy as np


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=False))


class StackedCBRBlock(nn.Sequential): 
    def __init__(self, in_c, out_c, num_blocks=1, kernel_size=3):
        assert num_blocks >= 1
        super().__init__()

        if kernel_size == 3:
            kernel_setting = dict(kernel_size=3, stride=1, padding=1)   
        elif kernel_size == 1:
            kernel_setting = dict(kernel_size=1)
        else:
            raise NotImplementedError

        cs = [in_c] + [out_c] * num_blocks    
        
        
        self.channel_pairs = self.slide_win_select(cs, win_size=2, win_stride=1, drop_last=True) 
        
        
        self.kernel_setting = kernel_setting

        for i, (i_c, o_c) in enumerate(self.channel_pairs):
            self.add_module(name=f"cbr_{i}", module=ConvBNReLU(i_c, o_c, **self.kernel_setting)) 

    @staticmethod
    def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
        num_items = len(items)
        i = 0
        while i + win_size <= num_items:
            yield items[i : i + win_size]
            i += win_stride

        if not drop_last:
            
            yield items[i : i + win_size]
    
    


class ConvFFN(nn.Module):
    def __init__(self, dim, out_dim=None, ffn_expand=4):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            StackedCBRBlock(dim, dim * ffn_expand, num_blocks=2, kernel_size=3),
            nn.Conv2d(dim * ffn_expand, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta
    
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = F.softmax(self.gn.gamma, dim=0)
        reweigts = self.sigomid(gn_x * w_gamma)
        
        info_mask = w_gamma > self.gate_treshold
        noninfo_mask = w_gamma <= self.gate_treshold
        x_1 = info_mask * reweigts * x
        x_2 = noninfo_mask * reweigts * x
        x = self.reconstruct(x_1, x_2)
        return x
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
class CRU(nn.Module):
    
    def __init__(self,
                 op_channel: int, 
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)        
        self.advavg = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1) 
        up, low = self.squeeze1(up), self.squeeze2(low) 
        
        Y1 = self.GWC(up) + self.PWC1(up) 
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        
        
        out = torch.cat([Y1, Y2], dim=1) 
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2
class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


    

class PatchwiseTokenReEmbedding:
    @staticmethod
    def encode(x, nh, ph, pw):
        return rearrange(x, "b (nh hd) (nhp ph) (nwp pw) -> b nh (hd ph pw) (nhp nwp)", nh=nh, ph=ph, pw=pw)   

    @staticmethod
    def decode(x, nhp, ph, pw):
        return rearrange(x, "b nh (hd ph pw) (nhp nwp) -> b (nh hd) (nhp ph) (nwp pw)", nhp=nhp, ph=ph, pw=pw) 



class GlobalContext(nn.Module):
    def __init__(
        self,
        channels,
        use_attn=True,
        fuse_add=False,
        fuse_scale=True,
        init_last_zero=False,
        rd_ratio=1.0 / 8,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer="sigmoid",
    ):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)
        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )
        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()
    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)
    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x
        return x


class SpatialViewAttn(nn.Module):
    def __init__(self, dim, p, nh=2): 
        super().__init__()
        self.p = p
        self.nh = nh
        self.scale = (dim // nh * self.p ** 2) ** -0.5    

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None, need_weights: bool = False):
        if kv is None:
            kv = q
        N, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)

        
        q = PatchwiseTokenReEmbedding.encode(q, nh=self.nh, ph=self.p, pw=self.p)
        k = PatchwiseTokenReEmbedding.encode(k, nh=self.nh, ph=self.p, pw=self.p)
        v = PatchwiseTokenReEmbedding.encode(v, nh=self.nh, ph=self.p, pw=self.p)

        qk = torch.einsum("bndx, bndy -> bnxy", q, k) * self.scale     
        qk = qk.softmax(-1)
        qkv = torch.einsum("bnxy, bndy -> bndx", qk, v) 

        qkv = PatchwiseTokenReEmbedding.decode(qkv, nhp=H // self.p, ph=self.p, pw=self.p)

        x = self.proj(qkv)
        if not need_weights:
            return x
        else:
            
            return x, qk.mean(dim=1)


class ChannelViewAttn(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        self.nh = nh
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        B, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)
        
        q = q.reshape(B, self.nh, C // self.nh, H * W)
        k = k.reshape(B, self.nh, C // self.nh, H * W)
        v = v.reshape(B, self.nh, C // self.nh, H * W)

        q = q * (q.shape[-1] ** (-0.5))
        qk = q @ k.transpose(-2, -1)
        qk = qk.softmax(dim=-1)
        qkv = qk @ v

        qkv = qkv.reshape(B, C, H, W)
        x = self.proj(qkv)
        return x

class SelfAttention(nn.Module):    
    def __init__(self, dim, p, nh, ffn_expand,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.sa = SpatialViewAttn(dim, p=p, nh=nh) 
        
        
        self.ca = ChannelViewAttn(dim, nh=nh)  
        
        self.alpha = nn.Parameter(data=torch.zeros(1)) 
        self.beta = nn.Parameter(data=torch.zeros(1)) 
        self.norm2 = nn.BatchNorm2d(dim)
        
        self.ffn = ScConv(op_channel=dim)

    def forward(self, x):
        normed_x = self.norm1(x)       
        x = x + self.alpha.sigmoid() * self.sa(normed_x) + self.beta.sigmoid() * self.ca(normed_x)         
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttention(nn.Module):    
    def __init__(self, dim, p, nh=4, ffn_expand=1):
        super().__init__()
        self.rgb_norm2 = nn.BatchNorm2d(dim)
        self.thermal_norm2 = nn.BatchNorm2d(dim)

        self.thermal_to_rgb_sa = SpatialViewAttn(dim, p=p, nh=nh)
        
        self.thermal_to_rgb_ca = ChannelViewAttn(dim, nh=nh)
        self.rgb_alpha = nn.Parameter(data=torch.zeros(1))
        self.rgb_beta = nn.Parameter(data=torch.zeros(1))

        self.rgb_to_thermal_sa = SpatialViewAttn(dim, p=p, nh=nh)
        self.rgb_to_thermal_ca = ChannelViewAttn(dim, nh=nh)
        self.thermal_alpha = nn.Parameter(data=torch.zeros(1))
        self.thermal_beta = nn.Parameter(data=torch.zeros(1))

        self.norm3 = nn.BatchNorm2d(2 * dim)
        self.ffn = ConvFFN(dim=2 * dim, ffn_expand=ffn_expand, out_dim=2 * dim)

    def forward(self, rgb, thermal):
        normed_rgb = self.rgb_norm2(rgb) 
        normed_thermal = self.thermal_norm2(thermal)
        transd_rgb = self.rgb_alpha.sigmoid() * self.thermal_to_rgb_sa(normed_rgb, normed_thermal) + self.rgb_beta.sigmoid() * self.thermal_to_rgb_ca(normed_rgb, normed_thermal)  
        rgb_rgbd = rgb + transd_rgb
        transd_thermal = self.thermal_alpha.sigmoid() * self.rgb_to_thermal_sa(normed_thermal, normed_rgb) + self.thermal_beta.sigmoid() * self.rgb_to_thermal_ca(normed_thermal, normed_rgb)   
        thermal_rgbd = thermal + transd_thermal
        rgbd = torch.cat([rgb_rgbd, thermal_rgbd], dim=1)
        
        return rgbd


class BIU(nn.Module):
    def __init__(self, in_dim, embed_dim, p, nh, ffn_expand):
        super().__init__()
        self.p = p
        self.rgb_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.thermal_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )

        self.rgb_imsa = SelfAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand) 
        self.thermal_imsa = SelfAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)
        self.imca = CrossAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)

        self.norm3 = nn.BatchNorm2d(2 * embed_dim)
        self.ffn = ConvFFN(dim=2 * embed_dim, ffn_expand=ffn_expand, out_dim=2 * embed_dim)
        self.cssa = SelfAttention(2 * embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)

    def forward(self, rgb, thermal, top_rgbd=None):
        
        rgb = self.rgb_cnn_proj(rgb) 
        thermal = self.thermal_cnn_proj(thermal) 

        rgb = self.rgb_imsa(rgb) 
        thermal = self.thermal_imsa(thermal)

        rgbd1 = self.imca(rgb, thermal)  
        if top_rgbd is not None:
            rgbd = rgbd1 + top_rgbd

        rgbd = rgbd1 + self.ffn(self.norm3(rgbd))
        return rgbd 
    





class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2)) 
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last 

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            
            out = F.relu(out, inplace=True)
            
        return out



class RSCU(torch.nn.Module):
    def __init__(self, channels):   
        super(RSCU, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2*channels, 128, 3, 1)

        self.conv_sum = ConvLayer(channels, 128, 3, 1)
        self.conv_mul = ConvLayer(channels, 128, 3, 1)

        block = []
        block += [ConvLayer(256, 128, 1, 1),
                  ConvLayer(128, 128, 3, 1),
                  ConvLayer(128, 128, 3, 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_vi, x_ir):  
        _, self.C, _, _ = x_vi.shape
        f_sum = x_ir + x_vi   
        f_mul = x_ir * x_vi   
        f_init = torch.cat([f_sum, f_sum], 1)   
        f_init = self.conv_fusion(f_init)  

        out_ir = self.conv_sum(f_sum)   
        out_vi = self.conv_mul(f_mul)   
        out = torch.cat([out_ir, out_vi], 1)   
        out = self.bottelblock(out)   

        out = f_init + out   
        
        return out 







class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class EDS(nn.Module):
    def __init__(self, in_channel=256, out_channel=128):
        super(EDS, self).__init__()

        self.conv1 = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(out_channel, int(out_channel / 4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(out_channel, int(out_channel / 4), kernel_size=3, dilation=3, padding=3) 
        self.dconv3 = BasicConv2d(out_channel, int(out_channel / 4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(out_channel, int(out_channel / 4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)


    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        fuse_x = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.fuse_dconv(fuse_x)

        return out
   
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias) 
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    


    

class Mlpp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Edge_Aware(nn.Module):
    def __init__(self, ):
        super(Edge_Aware, self).__init__()
        self.pos_embed1 = BasicConv2d(64, 64 ) 
        self.pos_embed2 = BasicConv2d(64, 64)
        self.conv31 = nn.Conv2d(64,1, kernel_size=1)
        self.conv128_64_x4 = TransBasicConv2d(128,64) 
        self.conv128_64_x3 = TransBasicConv2d(128, 64)
        self.conv128_64_x2 = TransBasicConv2d(128, 64)
        self.conv128_64_x1 = TransBasicConv2d(128, 64)
        self.up = nn.Upsample(56)
        self.up2 = nn.Upsample(256)  
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.BatchNorm2d(64)
        self.drop_path = DropPath(0.3)
        self.maxpool =nn.AdaptiveMaxPool2d(1)
        
        self.num_heads = 8
        self.mlp1 = Mlpp(in_features=64, out_features=64)
        self.mlp2 = Mlpp(in_features=64, out_features=64)
        self.mlp3 = Mlpp(in_features=64, out_features=64)
    def forward(self, x, y, z, v):   
        v = self.up(self.conv128_64_x4(v))  
        z = self.up(self.conv128_64_x3(z))  
        y = self.up(self.conv128_64_x2(y))  
        x = self.up(self.conv128_64_x1(x))    

        xx = x   
       

        x_max = self.maxpool(x)  
        
        b,_,_,_ = x_max.shape
        x_max = x_max.reshape(b, -1)
        
        x_y = self.mlp1(x_max)  
        
        x_z = self.mlp2(x_max)  
        x_v = self.mlp3(x_max)  

        x_y = x_y.reshape(b, 64, 1, 1)
        x_z = x_z.reshape(b, 64, 1, 1)
        x_v = x_v.reshape(b, 64, 1, 1) 
        

        x_y = torch.mul(x_y, y)   
        x_z = torch.mul(x_z, z)   
        x_v = torch.mul(x_v, v)

        x_mix =  x_v + x_z + x_y + xx  
        x_mix_1 = self.norm2(x_mix)
        x_mix_1 = self.pos_embed1(x_mix_1)  

        x_mix_2 = self.drop_path(x_mix_1)
        x_mix_2 = self. pos_embed2(x_mix_2)

        x_mix = x_mix_1 + x_mix_2
        x_mix = self.up2(self.conv31(x_mix)) 
    
        
        return x_mix



class BEINet_R101(nn.Module):
    def __init__(self, ps=(8, 8, 8, 8), embed_dim=64, pretrained=None): 
        super().__init__()
        self.rgb_encoder: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        ) 
        self.thermal_encoder: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        if pretrained: 
            self.rgb_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
            self.thermal_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)

        self.BIUs = nn.ModuleList(
            [
                BIU(in_dim=c, embed_dim=embed_dim, p=p, nh=2, ffn_expand=1) 
                for i, (p, c) in enumerate(zip(ps, (2048, 1024, 512, 256)))
            ]
        )  

        self.RSCU = RSCU(2048)  
        self.EDS = EDS(256, 128)  

        self.edge = Edge_Aware()
 
        self.predictor = nn.ModuleList()
        self.predictor.append(StackedCBRBlock(embed_dim * 2, embed_dim))
        self.predictor.append(StackedCBRBlock(embed_dim, 32))
        self.predictor.append(nn.Conv2d(32, 1, 1))

    def forward(self, data):
        rgb_feats = self.rgb_encoder(data["image"])
        thermal_feats = self.thermal_encoder(data["t"].repeat(1, 3, 1, 1)) 

        f_rgb4 = rgb_feats[3] 
        f_t4 = thermal_feats[3]
        x44 = self.RSCU(f_rgb4, f_t4)   
      
        x4 = self.BIUs[0](rgb=rgb_feats[3], thermal=thermal_feats[3], top_rgbd=x44)   
        x3 = self.BIUs[1](rgb=rgb_feats[2], thermal=thermal_feats[2], top_rgbd=cus_sample(x4, scale_factor=2))   
        x2 = self.BIUs[2](rgb=rgb_feats[1], thermal=thermal_feats[1], top_rgbd=cus_sample(x3, scale_factor=2))   
        x1 = self.BIUs[3](rgb=rgb_feats[0], thermal=thermal_feats[0], top_rgbd=cus_sample(x2, scale_factor=2))   
       

        x11 = self.EDS(rgb_feats[0], thermal_feats[0])

        x1_11 = x1 + x11 
        edge= self.edge(x1_11, x2, x3, x4)

        x1 = self.predictor[0](cus_sample(x1, scale_factor=2))         
  
        x1 = self.predictor[1](cus_sample(x1, scale_factor=2))  

        x1 = self.predictor[2](x1)


        x1_11 = self.predictor[0](cus_sample(x1_11, scale_factor=2))  
        x1_11 = self.predictor[1](cus_sample(x1_11, scale_factor=2))
        x1_11 = self.predictor[2](x1_11)

        x2 = cus_sample(x2, scale_factor=2)  
        x2 = self.predictor[0](cus_sample(x2, scale_factor=2))
        x2 = self.predictor[1](cus_sample(x2, scale_factor=2))
        x2 = self.predictor[2](x2)

        x3 = cus_sample(x3, scale_factor=4)  
        x3 = self.predictor[0](cus_sample(x3, scale_factor=2))
        x3 = self.predictor[1](cus_sample(x3, scale_factor=2))
        x3 = self.predictor[2](x3)

        x4 = cus_sample(x4, scale_factor=8)  
        x4 = self.predictor[0](cus_sample(x4, scale_factor=2))
        x4 = self.predictor[1](cus_sample(x4, scale_factor=2))
        x4 = self.predictor[2](x4)


        return x1_11, x1, x2, x3, x4, edge
    
