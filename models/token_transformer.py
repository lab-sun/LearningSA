# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take the standard Transformer as T2T Transformer
"""
import torch.nn as nn
from timm.models.layers import DropPath
from .transformer_block import Mlp
import torch

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim_medium = in_dim // num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        # self.q = nn.Linear(dim, in_dim)
        # self.k = nn.Linear(dim, in_dim)
        # self.v = nn.Linear(dim, in_dim)
        self.proj = nn.Linear(dim, in_dim)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(in_dim, in_dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.attn_gamma = nn.Parameter(torch.tensor(10.), requires_grad=True)

        self.sum_gamma0 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.sum_gamma1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # print('self.proj.parameters in token printed initial')
        # for para in self.proj.parameters():
        #     print(para)
        # os._exit()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + self.gamma*pos

    def L2normalize(self, x, d=3):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, x,pos=None):
        B, N, C = x.shape

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        if pos is not None:
            B_pos, C_pos, h_pos,w_pos = pos.shape
            pos=pos.reshape(B_pos,C_pos,-1).permute(0,2,1)

        # q=self.q(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        # k = self.k(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        # q = self.with_pos_embed(x, pos).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # k = self.with_pos_embed(x, pos).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # q = self.with_pos_embed(x, None).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # k = self.with_pos_embed(x, None).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q = self.proj(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        k = self.proj(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        v = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)#B, 1,N, dim

        q = self.L2normalize(q)
        k = self.L2normalize(k)

        attn = (q @ k.transpose(-2, -1)) / self.scale/self.attn_gamma
        # print('q',q)
        # print('k', k)
        # print('attn',attn)
        # attn=self.L2normalize(attn)
        # print('attn_l2', attn)

        attn = attn.softmax(dim=-1)

        # print('attn_after', attn)

        # print('torch.max(attn,3)',torch.max(attn,3))
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) #B,N, dim
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # # skip connection
        # x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        gamma_total = torch.exp(self.sum_gamma0) + torch.exp(self.sum_gamma1)

        # skip connection
        x = (torch.exp(self.sum_gamma0)/gamma_total)*x + (torch.exp(self.sum_gamma1)/gamma_total)*(v.transpose(1, 2).reshape(B, N, -1))  # because the original x has different size with current x, use v to do skip connection
        # print('token_self.gamma', self.gamma)
        # print('token_self.self.attn_gamma', self.attn_gamma)
        # print('token_torch.exp(self.sum_gamma0)/gamma_total', torch.exp(self.sum_gamma0) / gamma_total)
        # print('token_torch.exp(self.sum_gamma1)/gamma_total', torch.exp(self.sum_gamma1) / gamma_total)


        x= self.proj(x)
        # os._exit()


        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(in_dim)
        # self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,pos=None):
        # x = self.attn(self.norm1(x),pos)
        x = self.attn(x, pos)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention_interaction(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # self.in_dim_medium = in_dim // num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        # self.q = nn.Linear(dim, in_dim)
        # self.k = nn.Linear(dim, in_dim)
        # self.v = nn.Linear(dim, in_dim)
        # self.proj = nn.Linear(dim, in_dim)
        self.proj = nn.Linear(in_dim, in_dim)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(in_dim, in_dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # Learnable gain parameter
        # self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.attn_gamma = nn.Parameter(torch.tensor(10.), requires_grad=True)

        self.sum_gamma0 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.sum_gamma1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # print('self.proj.parameters in token printed initial')
        # for para in self.proj.parameters():
        #     print(para)
        # os._exit()

    # def with_pos_embed(self, tensor, pos):
    #     return tensor if pos is None else tensor + self.gamma*pos

    def L2normalize(self, x, d=3):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, x,y,pos=None):
        # B, H,N, C = x.shape

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # if pos is not None:
        #     B_pos, C_pos, h_pos,w_pos = pos.shape
        #     pos=pos.reshape(B_pos,C_pos,-1).permute(0,2,1)

        # q=self.q(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        # k = self.k(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        # q = self.with_pos_embed(x, pos).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # k = self.with_pos_embed(x, pos).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # q = self.with_pos_embed(x, None).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # k = self.with_pos_embed(x, None).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # q = self.proj(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        # k = self.proj(self.with_pos_embed(x, pos)).reshape(B, N, self.num_heads, self.in_dim_medium).permute(0, 2, 1, 3)
        # v = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)#B, 1,N, dim

        q = self.L2normalize(x) #(h*w)*b*4 *256
        k = self.L2normalize(x)

        print('q.shape',q.shape)

        attn = (q @ k.transpose(-2, -1)) / self.scale/self.attn_gamma  #(h*w)*b*4 *4

        print('attn',attn)
        print('attn.shape',attn.shape)
        # print('q',q)
        # print('k', k)
        # print('attn',attn)
        # attn=self.L2normalize(attn)
        # print('attn_l2', attn)

        attn = attn.softmax(dim=-1)

        print('attn',attn)
        print('attn.shape',attn.shape)

        # print('attn_after', attn)

        # print('torch.max(attn,3)',torch.max(attn,3))
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1) #B,N, dim
        y_add = attn @ y #(h*w)*b*4 *(h*w)
        print('y_add.shape',y_add.shape)
        y_d0,y_d1,y_d2,y_d3=y_add.shape 
        y_add=y_add.reshape(y_d0,y_d1*y_d2,y_d3)  #(h*w)*(b*4) *(h*w)
        print('y_add.shape',y_add.shape)
        y_add=self.proj(y_add).reshape(y_d0,y_d1,y_d2,y_d3)
        print('y_add.shape',y_add.shape)

        # x = self.proj(x)
        # x = self.proj_drop(x)

        # # skip connection
        # x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        gamma_total = torch.exp(self.sum_gamma0) + torch.exp(self.sum_gamma1)

        # skip connection
        # x = (torch.exp(self.sum_gamma0)/gamma_total)*x + (torch.exp(self.sum_gamma1)/gamma_total)*(v.transpose(1, 2).reshape(B, N, -1))  # because the original x has different size with current x, use v to do skip connection
        y = (torch.exp(self.sum_gamma0)/gamma_total)*y + (torch.exp(self.sum_gamma1)/gamma_total)*y_add  # because the original x has different size with current x, use v to do skip connection
        # print('token_self.gamma', self.gamma)
        # print('token_self.self.attn_gamma', self.attn_gamma)
        # print('token_torch.exp(self.sum_gamma0)/gamma_total', torch.exp(self.sum_gamma0) / gamma_total)
        # print('token_torch.exp(self.sum_gamma1)/gamma_total', torch.exp(self.sum_gamma1) / gamma_total)

        print('y.shape',y.shape)

        # x= self.proj(x)
        # os._exit()


        return y

class Token_transformer_interaction(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = Attention_interaction(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, proj_drop=drop, attn_drop=attn_drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(in_dim)
        # self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,y,pos=None):
        # x = self.attn(self.norm1(x),pos)
        y = self.attn(x, y)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return y






