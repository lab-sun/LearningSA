# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.generator import AdaptiveFeatureGenerator, DomainClassifier, ReverseLayerF
from util.util import vgg_preprocess
import util.util as util
import numpy as np
import os
import cv2
from models.transformer import build_transformer
from models.position_encoding import build_position_encoding
import torchvision.utils as vutils

from ..token_transformer import Token_transformer
from ..token_performer import Token_performer

def tensor2arrary_image(tgt_var_image):
    tgt_var_image_numpy=tgt_var_image.squeeze(0).permute(1,2,0).numpy()#b*c*h*w->h*w*c
    print('tgt_var_image_numpy',tgt_var_image_numpy)
    # tgt_var_image_numpy = (tgt_var_image_numpy + 1) / 2 * 255
    # print('tgt_var_image_numpy2', tgt_var_image_numpy)
    tgt_var_image_numpy_RGB = cv2.cvtColor(tgt_var_image_numpy, cv2.COLOR_BGR2RGB)#rgb-->bgr


    return tgt_var_image_numpy_RGB

class selfattention(nn.Module):
    def __init__(self, cin, cout,feature_H,feature_W):
        super(selfattention, self).__init__()
        # Channel multiplier
        self.cin = cin
        self.cout = cout
        # self.conv_q = nn.Sequential(
        #     nn.Conv2d(self.cin, self.cin, kernel_size=5, stride=1, padding=2),
        #     nn.InstanceNorm2d(self.cin),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cin, self.cout, kernel_size=5, stride=1, padding=2),
        #     nn.InstanceNorm2d(self.cout),
        #     nn.ReLU()
        # )
        self.conv_q = nn.Conv2d(self.cin, self.cout, kernel_size=1, stride=1, padding=0)
        self.conv_k = nn.Conv2d(self.cin, self.cout, kernel_size=1, stride=1, padding=0)
        self.conv_v = nn.Conv2d(self.cin, self.cout, kernel_size=1, stride=1, padding=0)
        # self.norm=nn.LayerNorm([cout,feature_H,feature_W])
        self.norm = nn.LayerNorm(cout)#normalize the last dimension
        # self.conv_k = nn.Sequential(
        #     nn.Conv2d(self.cin, self.cin, kernel_size=5, stride=1, padding=2),
        #     nn.InstanceNorm2d(self.cin),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cin, self.cout, kernel_size=5, stride=1, padding=2),
        #     nn.InstanceNorm2d(self.cout),
        #     nn.ReLU()
        # )
        # self.conv_v = nn.Sequential(
        #     nn.Conv2d(self.cin, self.cin, kernel_size=5, stride=1, padding=2),
        #     nn.InstanceNorm2d(self.cin),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cin, self.cout, kernel_size=5, stride=1, padding=2),
        #     nn.InstanceNorm2d(self.cout),
        #     nn.ReLU()
        # )
        # self.o = nn.Conv2d(self.cout, self.cin, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        # Apply convs
        q=self.conv_q(x)
        k=self.conv_k(x)
        v=self.conv_v(x)
        q = q.view(-1, self.cout, x.shape[2] * x.shape[3])
        k = k.view(-1, self.cout, x.shape[2] * x.shape[3])
        v = v.view(-1, self.cout, x.shape[2] * x.shape[3])

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(q.permute(0,2,1), k), -1)#b*(hq*wq)*(hk*wk)
        # Attention map times g path
        o = torch.bmm(v, beta.permute(0,2,1)).reshape(-1, self.cout, x.shape[2], x.shape[3])

        return self.norm(o + x)

class refattention(nn.Module):
    def __init__(self, cin, cout,feature_H,feature_W):
        super(refattention, self).__init__()
        # Channel multiplier
        self.cin = cin
        self.cout = cout
        self.conv_q = nn.Conv2d(self.cin, self.cout, kernel_size=1, stride=1, padding=0)
        self.conv_k = nn.Conv2d(self.cin, self.cout, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.LayerNorm(feature_H*feature_W) #norm channel

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.sum_gamma0 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.sum_gamma1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)



    def forward(self, x,v=None,pos=None):
        #pos
        if pos is not None:
            q=x+self.gamma*pos
            k = x + self.gamma * pos

        else:
            q=x.clone()
            k=x.clone()

        # # Apply convs
        # q = self.conv_q(x)
        # k = self.conv_k(x)

        # scale query
        scaling = float(self.cout) ** -0.5
        q = q * scaling

        q = self.L2normalize(q)
        k = self.L2normalize(k)

        q = q.view(-1, self.cout, x.shape[2] * x.shape[3])#b*c*(hq*wq)
        k = k.view(-1, self.cout, x.shape[2] * x.shape[3])#b*c*(hk*wk)
        v = v.view(-1, v.shape[1], v.shape[2] * v.shape[3])#b*cv*(hv*wv)


        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(q.permute(0, 2, 1), k), -1)  # b*(hq*wq)*(hk*wk)
        # Attention map times g path
        o = torch.bmm(v, beta.permute(0, 2, 1)).reshape(-1, v.shape[1], x.shape[2], x.shape[3]) # b*cv*hq*wq

        # print('q',q)
        # print('torch.bmm(q.permute(0, 2, 1), k)',torch.bmm(q.permute(0, 2, 1), k))
        # os._exit()



        # print('self.sum_gamma',self.sum_gamma)
        # print('self.sum_softmax(self.sum_gamma)', self.sum_softmax(self.sum_gamma))
        # self.sum_gamma.data = self.sum_softmax(self.sum_gamma)
        # print('self.sum_gamma[0]',self.sum_gamma[0])
        # print('self.sum_gamma[1]', self.sum_gamma[1])

        gamma_total=torch.exp(self.sum_gamma0)+torch.exp(self.sum_gamma1)

        out=(torch.exp(self.sum_gamma0)/gamma_total)*o + (torch.exp(self.sum_gamma1)/gamma_total)*(v.reshape(-1, v.shape[1], x.shape[2], x.shape[3]))
        # out=self.norm1(out.permute(0,2,3,1)) #norm the score maps makes limited influence on softmax

        # os._exit()
        print('self.gamma', self.gamma)
        print('torch.exp(self.sum_gamma0)/gamma_total', torch.exp(self.sum_gamma0)/gamma_total)
        print('torch.exp(self.sum_gamma1)/gamma_total', torch.exp(self.sum_gamma1) / gamma_total)

        # return out.permute(0,3,1,2)
        return  out,beta.reshape(-1, beta.shape[1], x.shape[2], x.shape[3])
        # return o

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            # self.attention1 = Token_transformer(dim=in_chans * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention1 = Token_transformer(dim=in_chans * 3 * 3, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0)
            # self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            # self.project = nn.Linear(token_dim * 3 * 3, embed_dim)
            # self.project2 = nn.Linear(embed_dim * 3 * 3, embed_dim)
            # self.project3 = nn.Linear(token_dim, embed_dim)
            # self.project3_1 = nn.Linear(in_chans * 3 * 3, token_dim)
            # self.project3_2 = nn.Linear(token_dim, embed_dim)

        # self.instnorm = nn.Sequential(
        #     nn.InstanceNorm2d(token_dim),
        #     nn.ReLU()
        # )
        # self.instnorm2 = nn.Sequential(
        #     nn.InstanceNorm2d(embed_dim),
        #     nn.ReLU()
        # )

    def forward(self, x,pos=None):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: restricturization/reconstruction
        x = self.attention1(x,pos)
        # B, new_HW, C = x.shape
        # x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # # iteration1: soft split
        # x = self.soft_split1(x).transpose(1, 2)
        #
        # # iteration2: restricturization/reconstruction
        # x = self.attention2(x)
        # B, new_HW, C = x.shape
        # x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # # iteration2: soft split
        # x = self.soft_split2(x).transpose(1, 2)
        #
        # # final tokens
        # x = self.project(x)

        # x = self.project3_2(self.project3_1(x))


        #reshape
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out

class WTA_scale(torch.autograd.Function):
    """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None

class VGG19_feature_color_torchversion(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        # print('x',x)
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class find_correspondence(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(find_correspondence, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma

        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
                                               np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).cuda()
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).cuda()

        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,1,3).cuda()
        self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
                                                                                                                  3,
                                                                                                                  1).cuda()

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).cuda()
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).cuda()

        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1, 1, feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).cuda()
        self.y_normal = np.linspace(-1, 1, feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).cuda()

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def softmax_with_temperature(self, x, beta, d=1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(beta * x)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum


    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def kernel_soft_argmax(self, corr):
        b, _, h, w = corr.size()
        # corr = self.apply_gaussian_kernel(corr, sigma = self.kernel_sigma)

        corr = self.softmax_with_temperature(corr, beta=self.beta, d=1)

        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x.cuda() * x_normal.cuda()).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y.cuda() * y_normal.cuda()).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    def get_flow_smoothness(self, flow, GT_mask):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx) * GT_mask  # consider foreground regions only
        flow_dy = torch.abs(flow_dy) * GT_mask

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def get_flow_smoothness_nomask(self, flow):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel.cuda()) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel.cuda()) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx)  # consider whole regions
        flow_dy = torch.abs(flow_dy)

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def forward(self, corr, GT_mask=None):
        b, _, h, w = corr.size()
        grid_X = self.grid_X.expand(b, h, w)  # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w)  # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)

        if self.beta is not None:
            grid_x, grid_y = self.kernel_soft_argmax(corr)
        else:  # discrete argmax
            _, idx = torch.max(corr, dim=1)
            grid_x = idx % w
            grid_x = (grid_x.float() / (w - 1) - 0.5) * 2
            grid_y = idx // w
            grid_y = (grid_y.float() / (h - 1) - 0.5) * 2

        grid = torch.cat((grid_x.permute(0, 2, 3, 1), grid_y.permute(0, 2, 3, 1)),
                         3)  # 2-channels@3rd-dim, first channel for x / second channel for y
        flow = torch.cat((grid_x.cuda() - grid_X.cuda(), grid_y.cuda() - grid_Y.cuda()),
                         1)  # 2-channels@1st-dim, first channel for x / second channel for y

        if GT_mask is None:  # test
            smoothness = self.get_flow_smoothness_nomask(flow)
            return grid, flow, smoothness
        else:  # train
            smoothness = self.get_flow_smoothness(flow, GT_mask)
            return grid, flow, smoothness


class flow2grid_smooth(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(flow2grid_smooth, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma

        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
                                               np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).cuda()
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).cuda()

        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,1,3).cuda()
        self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
                                                                                                                  3,
                                                                                                                  1).cuda()


    def get_flow_smoothness_nomask(self, flow):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx)  # consider whole regions
        flow_dy = torch.abs(flow_dy)

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def forward(self, flow):
        b, _, h, w = flow.size()
        grid_X = self.grid_X.expand(b, h, w)  # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w)  # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)

        grid = torch.cat(((flow[:,0,:,:].unsqueeze(1)+grid_X).permute(0, 2, 3, 1), (flow[:,1,:,:].unsqueeze(1)+grid_Y).permute(0, 2, 3, 1)),
                         3)  # 2-channels@3rd-dim, first channel for x / second channel for y
        # grid = torch.cat((grid_x.permute(0, 2, 3, 1), grid_y.permute(0, 2, 3, 1)),
        #                  3)  # 2-channels@3rd-dim, first channel for x / second channel for y
        # flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),
        #                  1)  # 2-channels@1st-dim, first channel for x / second channel for y


        smoothness = self.get_flow_smoothness_nomask(flow)
        return grid, flow, smoothness


class find_correspondence_average(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(find_correspondence_average, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma

        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
                                               np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).cuda()
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).cuda()

        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,1,3).cuda()
        self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
                                                                                                                  3,
                                                                                                                  1).cuda()

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).cuda()
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).cuda()

        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1, 1, feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).cuda()
        self.y_normal = np.linspace(-1, 1, feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).cuda()

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def softmax_with_temperature(self, x, beta, d=1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(beta * x)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum


    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def kernel_soft_argmax(self, corr):
        b, _, h, w = corr.size()
        # corr = self.apply_gaussian_kernel(corr, sigma = self.kernel_sigma)

        # corr = self.softmax_with_temperature(corr, beta=self.beta, d=1)

        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    def get_flow_smoothness(self, flow, GT_mask):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx) * GT_mask  # consider foreground regions only
        flow_dy = torch.abs(flow_dy) * GT_mask

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def get_flow_smoothness_nomask(self, flow):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx)  # consider whole regions
        flow_dy = torch.abs(flow_dy)

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def forward(self, corr, GT_mask=None):
        b, _, h, w = corr.size()
        grid_X = self.grid_X.expand(b, h, w)  # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w)  # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)

        if self.beta is not None:
            grid_x, grid_y = self.kernel_soft_argmax(corr)
        else:  # discrete argmax
            _, idx = torch.max(corr, dim=1)
            grid_x = idx % w
            grid_x = (grid_x.float() / (w - 1) - 0.5) * 2
            grid_y = idx // w
            grid_y = (grid_y.float() / (h - 1) - 0.5) * 2

        grid = torch.cat((grid_x.permute(0, 2, 3, 1), grid_y.permute(0, 2, 3, 1)),
                         3)  # 2-channels@3rd-dim, first channel for x / second channel for y
        flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),
                         1)  # 2-channels@1st-dim, first channel for x / second channel for y

        if GT_mask is None:  # test
            smoothness = self.get_flow_smoothness_nomask(flow)
            return grid, flow, smoothness
        else:  # train
            smoothness = self.get_flow_smoothness(flow, GT_mask)
            return grid, flow, smoothness

class apply_gaussian_kernel(nn.Module):
    def __init__(self, feature_H, feature_W, kernel_sigma):
        super(apply_gaussian_kernel, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_sigma = kernel_sigma

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).cuda()
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).cuda()


    def apply_gaussian_kernel_detail(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def kernel_soft_argmax(self, corr):
        corr = self.apply_gaussian_kernel_detail(corr, sigma=self.kernel_sigma)
        return corr

    def forward(self, corr):
        corr= self.kernel_soft_argmax(corr)
        return corr

class matching_layer(nn.Module):
    def __init__(self):
        super(matching_layer, self).__init__()
        self.relu = nn.ReLU()

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        feature1 = feature1.view(b, c, h1 * w1)
        feature2 = feature2.view(b, c, h2 * w2)
        corr = torch.bmm(feature2.transpose(1, 2), feature1)
        corr = corr.view(b, h2 * w2, h1, w1)  # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr

class weights_gen(nn.Module):
    def __init__(self,in_c, out_c):
        super(weights_gen, self).__init__()
        self.conv1=nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(1024*20*20, 5)
        # self.fc2 = nn.Linear(5, 5)
        self.relu = nn.ReLU()
        self.soft=nn.Softmax(dim=1)


    def forward(self, feat_map):
        b, c, h, w = feat_map.size()
        # feat_map=feat_map.reshape(b,c*h*w)
        # weights_temp=self.relu(self.fc1(feat_map))
        # weights=self.soft(self.fc2(weights_temp))
        weights_temp=self.relu(self.conv1(feat_map))
        weights = self.soft(self.conv2(weights_temp))
        weights=weights.reshape(b, -1, h*w)
        return weights

class agg_corr(nn.Module):
    def __init__(self):
        super(agg_corr, self).__init__()

    def shiftVol(self, volume, shift_x=0, shift_y=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b, c, h, w = volume.size()
        if shift_x < 0:  # left shift,rightvolume
            theta = torch.tensor([[1, 0, (2.0 / (w - 1)) * (-shift_x)], [0, 1, 0]],
                                 dtype=torch.float32)  # x= x + 2.0 / 19  left shift
            volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)
            volume_output_revert[:, :, (w + shift_x):w, :, :] = volume_output_revert[:, :,
                                                                (w + shift_x - 1):(w + shift_x), :,
                                                                :]  # right margin padding
            volume_second = volume_output_revert.reshape(b, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # left shift for computing volume aggregation
            volume[:, :, :, (w + shift_x):w] = volume[:, :, :,
                                               (w + shift_x - 1):(w + shift_x)]  # right margin padding for volume
        if shift_x > 0:  # right shift,leftvolume
            theta = torch.tensor([[1, 0, (2.0 / (w - 1)) * (-shift_x)], [0, 1, 0]],
                                 dtype=torch.float32)  # x= x - 2.0 / 19  right shift
            volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)
            volume_output_revert[:, :, 0:shift_x, :, :] = volume_output_revert[:, :, shift_x:(shift_x + 1), :,
                                                          :]  # left margin padding
            volume_second = volume_output_revert.reshape(b, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # right shift for computing volume aggregation
            volume[:, :, :, 0:shift_x] = volume[:, :, :, shift_x:(shift_x + 1)]  # left margin padding for volume
        if shift_y > 0:  # down shift,upvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h - 1)) * (-shift_y)]],
                                 dtype=torch.float32)  # y= y - 2.0 / 19  down shift
            volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)  # b * h * w * h * w
            volume_output_revert[:, 0:shift_y, :, :, :] = volume_output_revert[:, shift_y:(shift_y + 1), :, :,
                                                          :]  # up margin padding
            volume_second = volume_output_revert.reshape(b, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # down shift for computing volume aggregation
            volume[:, :, 0:shift_y, :] = volume[:, :, shift_y:(shift_y + 1), :]  # up margin padding for volume
        if shift_y < 0:  # up shift,downvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h - 1)) * (-shift_y)]],
                                 dtype=torch.float32)  # y= y + 2.0 / 19  up shift
            volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)  # b * h' * w' * h * w
            volume_output_revert[:, (w + shift_y):w, :, :, :] = volume_output_revert[:, (w + shift_y - 1):(w + shift_y),
                                                                :, :, :]  # down margin padding
            volume_second = volume_output_revert.reshape(b, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # up shift for computing volume aggregation
            volume[:, :, (w + shift_y):w, :] = volume[:, :, (w + shift_y - 1):(w + shift_y),
                                               :]  # down margin padding for volume
        return volume

    def forward(self, corr,weights):
        b, c, h, w = corr.size()
        corr_S2T_right = self.shiftVol(corr, shift_x=-1, shift_y=0)
        corr_S2T_right = corr_S2T_right.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T_right_up = self.shiftVol(corr, shift_x=-1, shift_y=1)
        corr_S2T_right_up = corr_S2T_right_up.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T_left = self.shiftVol(corr, shift_x=1, shift_y=0)
        corr_S2T_left = corr_S2T_left.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T_left_up = self.shiftVol(corr, shift_x=1, shift_y=1)
        corr_S2T_left_up = corr_S2T_left_up.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T_up = self.shiftVol(corr, shift_x=0, shift_y=1)
        corr_S2T_up = corr_S2T_up.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T_down = self.shiftVol(corr, shift_x=0, shift_y=-1)
        corr_S2T_down = corr_S2T_down.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T_left_down = self.shiftVol(corr, shift_x=1, shift_y=-1)#sign represents the direction of the movement
        corr_S2T_left_down = corr_S2T_left_down.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr_S2T__right_down = self.shiftVol(corr, shift_x=-1, shift_y=-1)
        corr_S2T__right_down = corr_S2T__right_down.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr = corr.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        # 5layers for each pixel
        corr_cat = torch.cat((corr_S2T_up, corr_S2T_right_up, corr_S2T_right, corr_S2T__right_down, corr,
                              corr_S2T_down, corr_S2T_left_down, corr_S2T_left, corr_S2T_left_up),
                             1)  # b * 9 *  (h * w) * h' * w' clockwise like a hand of a clock

        weights_exten=weights.unsqueeze(3).unsqueeze(4)#b*9*400 -> b*9*400*1*1
        corr=corr_cat*weights_exten
        corr=torch.sum(corr,1)# b *  (h * w) * h' * w'
        corr = corr.reshape(-1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(
            -1, (h * w), h, w)  # b * (h' * w') * h * w
        return corr

class agg_flow(nn.Module):
    def __init__(self):
        super(agg_flow, self).__init__()

    def shiftVol(self, volume, shift_x=0, shift_y=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b, c, h, w = volume.size()
        if shift_x < 0:  # left shift,rightvolume
            theta = torch.tensor([[1, 0, (2.0 / (w - 1)) * (-shift_x)], [0, 1, 0]],
                                 dtype=torch.float32)  # x= x + 2.0 / 19  left shift
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume.size())
            grid = grid.to(device)
            volume = F.grid_sample(volume, grid)
            # volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)
            # volume_output_revert[:, :, (w + shift_x):w, :, :] = volume_output_revert[:, :,
            #                                                     (w + shift_x - 1):(w + shift_x), :,
            #                                                     :]  # right margin padding
            # volume_second = volume_output_revert.reshape(b, h * w, h, w)
            # volume = F.grid_sample(volume_second, grid)  # left shift for computing volume aggregation
            volume[:, :, :, (w + shift_x):w] = volume[:, :, :,
                                               (w + shift_x - 1):(w + shift_x)]  # right margin padding for volume
        if shift_x > 0:  # right shift,leftvolume
            theta = torch.tensor([[1, 0, (2.0 / (w - 1)) * (-shift_x)], [0, 1, 0]],
                                 dtype=torch.float32)  # x= x - 2.0 / 19  right shift
            # volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume.size())
            grid = grid.to(device)
            volume = F.grid_sample(volume, grid)
            # volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)
            # volume_output_revert[:, :, 0:shift_x, :, :] = volume_output_revert[:, :, shift_x:(shift_x + 1), :,
            #                                               :]  # left margin padding
            # volume_second = volume_output_revert.reshape(b, h * w, h, w)
            # volume = F.grid_sample(volume_second, grid)  # right shift for computing volume aggregation
            volume[:, :, :, 0:shift_x] = volume[:, :, :, shift_x:(shift_x + 1)]  # left margin padding for volume
        if shift_y > 0:  # down shift,upvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h - 1)) * (-shift_y)]],
                                 dtype=torch.float32)  # y= y - 2.0 / 19  down shift
            # volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume.size())
            grid = grid.to(device)
            volume = F.grid_sample(volume, grid)
            # volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)  # b * h * w * h * w
            # volume_output_revert[:, 0:shift_y, :, :, :] = volume_output_revert[:, shift_y:(shift_y + 1), :, :,
            #                                               :]  # up margin padding
            # volume_second = volume_output_revert.reshape(b, h * w, h, w)
            # volume = F.grid_sample(volume_second, grid)  # down shift for computing volume aggregation
            volume[:, :, 0:shift_y, :] = volume[:, :, shift_y:(shift_y + 1), :]  # up margin padding for volume
        if shift_y < 0:  # up shift,downvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h - 1)) * (-shift_y)]],
                                 dtype=torch.float32)  # y= y + 2.0 / 19  up shift
            # volume_temp = volume.reshape(b, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0).expand(b,2,3), volume.size())
            grid = grid.to(device)
            volume = F.grid_sample(volume, grid)
            # volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(b, h, w, h, w)  # b * h' * w' * h * w
            # volume_output_revert[:, (w + shift_y):w, :, :, :] = volume_output_revert[:, (w + shift_y - 1):(w + shift_y),
            #                                                     :, :, :]  # down margin padding
            # volume_second = volume_output_revert.reshape(b, h * w, h, w)
            # volume = F.grid_sample(volume_second, grid)  # up shift for computing volume aggregation
            volume[:, :, (w + shift_y):w, :] = volume[:, :, (w + shift_y - 1):(w + shift_y),
                                               :]  # down margin padding for volume
        return volume

    def forward(self, corr,weights):
        b, c, h, w = corr.size()
        corr_S2T_right = self.shiftVol(corr, shift_x=-1, shift_y=0)
        corr_S2T_right = corr_S2T_right.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T_right_up = self.shiftVol(corr, shift_x=-1, shift_y=1)
        corr_S2T_right_up = corr_S2T_right_up.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T_left = self.shiftVol(corr, shift_x=1, shift_y=0)
        corr_S2T_left = corr_S2T_left.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T_left_up = self.shiftVol(corr, shift_x=1, shift_y=1)
        corr_S2T_left_up = corr_S2T_left_up.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T_up = self.shiftVol(corr, shift_x=0, shift_y=1)
        corr_S2T_up = corr_S2T_up.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T_down = self.shiftVol(corr, shift_x=0, shift_y=-1)
        corr_S2T_down = corr_S2T_down.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T_left_down = self.shiftVol(corr, shift_x=1, shift_y=-1)#sign represents the direction of the movement
        corr_S2T_left_down = corr_S2T_left_down.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr_S2T__right_down = self.shiftVol(corr, shift_x=-1, shift_y=-1)
        corr_S2T__right_down = corr_S2T__right_down.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        corr = corr.unsqueeze(1).reshape(b, 1, -1, h*w).permute(0, 1, 3,2) # b * c *  (h * w) * 2
        # 5layers for each pixel
        corr_cat = torch.cat((corr_S2T_up, corr_S2T_right_up, corr_S2T_right, corr_S2T__right_down, corr,
                              corr_S2T_down, corr_S2T_left_down, corr_S2T_left, corr_S2T_left_up),
                             1)  # b * 9 *  (h * w) * 2 clockwise like a hand of a clock

        weights_exten=weights.unsqueeze(3)#b*9*400 -> b*9*400*1
        corr=corr_cat*weights_exten
        corr=torch.sum(corr,1)# b *  (h * w) * 2
        corr = corr.reshape(b, h, w, 2).permute(0, 3, 1, 2)# b * 2 * h * w
        return corr

class apply_kernel(nn.Module):
    def __init__(self, feature_H, feature_W):
        super(apply_kernel, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).cuda()
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).cuda()

    def kernel_detail(self, corr):

        # start=time.time()
        b, hw, h, w = corr.size()
        zero = torch.zeros_like(corr)
        one = torch.ones_like(corr)

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        # gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        # gauss_kernel = gauss_kernel.view(b, hw, h, w)
        kernel = ((x - idx_x) ** 2 + (y - idx_y) ** 2)
        # kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * 5 ** 2))
        # print(torch.max(kernel))
        # print(torch.min(kernel))
        kernel = kernel.view(b, hw, h, w)

        # end_kernel = time.time()

        # print('kernel', kernel)
        kernel= torch.where(kernel > 3, zero, one)
        # print('kernel_final',kernel.permute(0, 2, 3, 1).reshape(b, hw, hw).reshape(b, hw, h, w))

        # end_kernel_final=time.time()
        # print('end_kernel',end_kernel-start)
        # print('end_kernel_final', end_kernel_final-end_kernel)
        # print('kernel_sum',torch.sum(kernel, 1, out=None))

        return kernel * corr

    def kernel(self, corr):
        corr = self.kernel_detail(corr)
        return corr

    def forward(self, corr):
        corr= self.kernel(corr)
        return corr


class apply_kernel_separate(nn.Module):
    def __init__(self, c_H, c_W):
        super(apply_kernel_separate, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, c_W - 1, c_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).cuda()
        self.y = np.linspace(0, c_H - 1, c_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).cuda()

        self.c_h=c_H
        self.c_w = c_W

    def kernel_detail(self, corr):

        # start=time.time()
        b, hw, h, w = corr.size()
        zero = torch.zeros_like(corr)
        one = torch.ones_like(corr)

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // self.c_w).view(b, 1, 1, h, w).float()
        idx_x = (idx % self.c_w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, self.c_w, 1, 1).expand(b, 1, self.c_w, h, w)
        y = self.y.view(1, self.c_h, 1, 1, 1).expand(b, self.c_h, 1, h, w)

        # gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        # gauss_kernel = gauss_kernel.view(b, hw, h, w)
        kernel = ((x - idx_x) ** 2 + (y - idx_y) ** 2)
        # kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * 5 ** 2))
        # print(torch.max(kernel))
        # print(torch.min(kernel))
        kernel = kernel.view(b, hw, h, w)

        # end_kernel = time.time()

        # print('kernel', kernel)
        kernel= torch.where(kernel > 3, zero, one)
        # print('kernel_final',kernel.permute(0, 2, 3, 1).reshape(b, hw, hw).reshape(b, hw, h, w))

        # end_kernel_final=time.time()
        # print('end_kernel',end_kernel-start)
        # print('end_kernel_final', end_kernel_final-end_kernel)
        # print('kernel_sum',torch.sum(kernel, 1, out=None))

        return kernel * corr

    def kernel(self, corr):
        corr = self.kernel_detail(corr)
        return corr

    def forward(self, corr):
        corr= self.kernel(corr)
        return corr

class NoVGGCorrespondence(BaseNetwork):
    # input is Al, Bl, channel = 1, range~[0,255]
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        opt.spade_ic = opt.semantic_nc
        self.adaptive_model_seg = AdaptiveFeatureGenerator(opt)
        opt.spade_ic = 3
        self.adaptive_model_img = AdaptiveFeatureGenerator(opt)
        del opt.spade_ic
        if opt.weight_domainC > 0 and (not opt.domain_rela):
            self.domain_classifier = DomainClassifier(opt)

        if 'down' not in opt:
            opt.down = 4
        if opt.warp_stride == 2:
            opt.down = 2
        assert (opt.down == 2) or (opt.down == 4)

        opt.down = 8  # original scale

        self.down = opt.down
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256

        
        coord_c = 3 if opt.use_coordconv else 0
        label_nc = opt.semantic_nc if opt.maskmix else 0

        # self.layer = nn.Sequential(
        #     ResidualBlock(label_nc + coord_c, label_nc + coord_c,kernel_size=3, padding=1, stride=1),
        #     ResidualBlock(label_nc + coord_c, label_nc + coord_c,kernel_size=3, padding=1, stride=1),
        #     ResidualBlock(label_nc + coord_c, label_nc + coord_c,kernel_size=3, padding=1, stride=1),
        #     ResidualBlock(label_nc + coord_c, label_nc + coord_c,kernel_size=3, padding=1, stride=1))
    
        # self.layer = nn.Sequential(
        #     ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
        #     ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
        #     ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
        #     ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1))



        # self.phi = nn.Conv2d(in_channels=self.in_channels + label_nc + coord_c, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        # self.theta = nn.Conv2d(in_channels=self.in_channels + label_nc + coord_c, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(label_nc + coord_c, label_nc + coord_c, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm2d(label_nc + coord_c),
        #     nn.ReLU()
        # )

        self.instnorm = nn.Sequential(
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.instnorm2 = nn.Sequential(
            nn.InstanceNorm2d(label_nc + coord_c),
            nn.ReLU()
        )


        self.upsampling_bi = nn.Upsample(scale_factor=opt.down, mode='bilinear') #for show
        if opt.warp_bilinear:
            self.upsampling = nn.Upsample(scale_factor=opt.down, mode='bilinear')
        else:
            self.upsampling = nn.Upsample(scale_factor=opt.down)
        self.zero_tensor = None

        self.feature_H=32
        self.feature_W = 32
        self.weight_c=9

        # self.selfatten = selfattention(label_nc + coord_c, label_nc + coord_c,self.feature_H,self.feature_W)
        self.refatten = refattention(label_nc + coord_c, label_nc + coord_c,self.feature_H,self.feature_W)



        self.conv_corr = nn.Sequential(
            nn.Conv2d(self.feature_H * self.feature_W, self.feature_H, kernel_size=3, stride=1,
                      padding=1),
            nn.Conv2d(self.feature_H, 2, kernel_size=3, stride=1,padding=1),
            nn.ReLU()
        )

        self.find_correspondence = find_correspondence(feature_H=self.feature_H, feature_W=self.feature_W, beta=500, kernel_sigma=5)
        self.flow2grid_smooth = flow2grid_smooth(feature_H=self.feature_H, feature_W=self.feature_W, beta=100,
                                                       kernel_sigma=5)
        self.find_correspondence_average = find_correspondence_average(feature_H=self.feature_H, feature_W=self.feature_W, beta=100, kernel_sigma=5)
        self.apply_kernel=apply_kernel(feature_H=self.feature_H,feature_W=self.feature_W)
        self.apply_kernel_separate=apply_kernel_separate(c_H=self.feature_H*opt.down,c_W=self.feature_W*opt.down)
        self.matching_layer = matching_layer()
        self.weights_gen=weights_gen(self.inter_channels,self.weight_c)
        self.agg_corr=agg_corr()
        self.agg_flow=agg_flow()
        self.apply_gaussian_kernel=apply_gaussian_kernel(feature_H=self.feature_H, feature_W=self.feature_W, kernel_sigma=5)

        self.tokens_to_token = T2T_module(tokens_type='transformer', in_chans=label_nc + coord_c, embed_dim=64)
        self.tokens_to_token2 = T2T_module(tokens_type='transformer', in_chans=64,
                                          embed_dim=label_nc + coord_c)
        # self.transformer = build_transformer(opt)
        self.position_embedding = build_position_encoding(opt,scale=3*3)
        self.position_embedding2 = build_position_encoding(opt)

        # self.transformer_ref = build_transformer_ref(opt,self.feature_H)



    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
        
        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def shift_warp(self, grid_S2T,ref_img,shift_x,shift_y):
        #b*256*256*2 full resolution
        # print('grid_S2T1', grid_S2T)
        _,h,w,_=grid_S2T.size()
        grid_S2T_up_x = grid_S2T[:, :, :, 0] + shift_x * 2.0 / (w - 1)
        grid_S2T_up_y = grid_S2T[:, :, :, 1] + shift_y * 2.0 / (h - 1)
        grid_S2T_up = torch.cat((grid_S2T_up_x.unsqueeze(3), grid_S2T_up_y.unsqueeze(3)), 3)
        # print('grid_S2T2', grid_S2T_up)
        grid_S2T_up = torch.clamp(grid_S2T_up, -1.0, 1.0, out=None)  # clamp into -1 to 1

        # print('grid_S2T3', grid_S2T_up)

        warped_tgt_up = F.grid_sample(ref_img.cuda(), grid_S2T_up,
                                      mode='bilinear')  # warping the colored original image

        return warped_tgt_up

    def find_smlr(self, corr_S2T,max_idx_y,max_idx_x,idx_y,idx_x):
        b,_,h,w=corr_S2T.size()
        corr_S2T_temp = corr_S2T.reshape(b, h,w, h,w)
        for i in range(b):
            if i == 0:
                max_smlr = corr_S2T_temp[
                    i, max_idx_y[i, :, :], max_idx_x[i, :, :], idx_y[i, :, :], idx_x[i, :, :]].unsqueeze(0)
            else:
                max_smlr = torch.cat((max_smlr, corr_S2T_temp[
                    i, max_idx_y[i, :, :], max_idx_x[i, :, :], idx_y[i, :, :], idx_x[i, :, :]].unsqueeze(0)), 0)

        return max_smlr

    def forward(self,
                ref_img,
                real_img,
                seg_map,
                ref_semantics_r, ref_semantics_g, ref_semantics_b,
                ref_seg_map,
                temperature=0.01,
                detach_flag=False,
                WTA_scale_weight=1,
                alpha=1,
                return_corr=False,
                photo=None,photo_semantics=None,
                input_label=None,ref_no_norm=None,
                logs_writer=None,batch_i=0):
        coor_out = {}
        batch_size = ref_img.shape[0]
        image_height = ref_img.shape[2]
        image_width = ref_img.shape[3]
        feature_height = int(image_height / self.opt.down)
        feature_width = int(image_width / self.opt.down)

        # if self.opt.mask_noise: #add noise to mask
        #     noise = torch.randn_like(seg_map, requires_grad=False) * 0.1
        #     noise[seg_map == 0] = 0
        #     seg_input = seg_map + noise
        # else:
        #     seg_input = seg_map
        # # print('seg_input.size',seg_input.size())
        # adaptive_feature_seg = self.adaptive_model_seg(seg_input, seg_input)
        # adaptive_feature_img = self.adaptive_model_img(ref_img, ref_img)
        # adaptive_feature_seg = util.feature_normalize(adaptive_feature_seg)
        # adaptive_feature_img = util.feature_normalize(adaptive_feature_img)
        # if self.opt.isTrain and self.opt.novgg_featpair > 0:
        #     adaptive_feature_img_pair = self.adaptive_model_img(real_img, real_img)
        #     adaptive_feature_img_pair = util.feature_normalize(adaptive_feature_img_pair)
        #     coor_out['loss_novgg_featpair'] = F.l1_loss(adaptive_feature_seg, adaptive_feature_img_pair) * self.opt.novgg_featpair
        #
        # if self.opt.use_coordconv:
        #     adaptive_feature_seg = self.addcoords(adaptive_feature_seg)
        #     adaptive_feature_img = self.addcoords(adaptive_feature_img)

        # print('adaptive_feature_seg.size()',adaptive_feature_seg.size())
        seg = F.interpolate(seg_map, size=(feature_height,feature_width), mode='nearest')
        ref_seg = F.interpolate(ref_seg_map, size=(feature_height,feature_width), mode='nearest')
        photo_seg = F.interpolate(photo_semantics, size=(feature_height,feature_width), mode='nearest')


        # print('seg.shape',seg.shape)

        # if self.opt.maskmix:
        #     cont_features = self.layer(torch.cat((adaptive_feature_seg, seg), 1))
        #     if self.opt.noise_for_mask and ((not self.opt.isTrain) or (self.opt.isTrain and self.opt.epoch > self.opt.mask_epoch)):
        #         # print('noise')
        #         noise = torch.randn_like(ref_seg, requires_grad=False) * 0.01
        #         ref_features = self.layer(torch.cat((adaptive_feature_img, noise), 1))
        #     else:
        #         # print('no noise')
        #         ref_features = self.layer(torch.cat((adaptive_feature_img, ref_seg), 1))
        #
        #     # cont_features = self.layer(seg)
        #     # ref_features = self.layer(ref_seg)
        #
        # else:
        #     #never go through ....  chanel number input of self.layer is self.feature_channel * 4 + label_nc + coord_c
        #     cont_features = self.layer(adaptive_feature_seg)
        #     ref_features = self.layer(adaptive_feature_img)
        #
        # # os._exit()

        similarity_computation_mode='SANet'
        wether_shrink_corr_agg=False#something wrong
        direct_corr2flow = False
        corr_agg=False
        flow_agg=False
        if similarity_computation_mode=='SANet':

            # # self-attention
            # theta = self.selfatten(seg)  # src feature
            # phi = self.selfatten(ref_seg)  # target feature
            # gama = self.selfatten(photo_seg)  # photo feature

            pos=self.position_embedding(seg)

            theta = seg # src feature
            phi = ref_seg # target feature
            gama = photo_seg # photo feature
            theta_b, theta_c, theta_h, theta_w = theta.size()

            if not self.opt.isTrain:
                feat = theta[0, :, :, :].unsqueeze(dim=1)
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('input_feat', feat_grid, global_step=batch_i)



            # theta = self.instnorm(self.tokens_to_token(self.instnorm(theta)))
            # phi = self.instnorm(self.tokens_to_token(self.instnorm(phi)))
            # gama = self.instnorm(self.tokens_to_token(self.instnorm(gama)))

            # print('theta.shape',theta.shape)
            theta = self.instnorm(self.tokens_to_token(theta,pos))
            phi = self.instnorm(self.tokens_to_token(phi,pos))
            gama = self.instnorm(self.tokens_to_token(gama,pos))

            # print('theta.shape', theta.shape)
            theta = self.instnorm2(self.tokens_to_token2(theta))
            phi = self.instnorm2(self.tokens_to_token2(phi))
            gama = self.instnorm2(self.tokens_to_token2(gama))

            # print('theta.shape', theta.shape)
            # os._exit()

            pos = self.position_embedding2(seg)

            # os._exit()
            # theta = self.instnorm(self.transformer(theta, None, pos))  # src feature
            # phi = self.instnorm(self.transformer(phi, None, pos))  # target feature
            # gama = self.instnorm(self.transformer(gama, None, pos))  # photo feature

            if not self.opt.isTrain:
                feat = theta[0, :, :, :].unsqueeze(dim=1)
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('after_tr_feat', feat_grid, global_step=batch_i)


            # adaptation
            theta_adap = theta # src feature
            phi_adap = phi  # target feature
            gama_adap = gama  # photo feature
            # theta_adap = self.conv1(theta)  # src feature
            # phi_adap = self.conv1(phi)  # target feature
            # gama_adap = self.conv1(gama)  # photo feature


            if not self.opt.isTrain:
                feat = theta_adap[0, :, :, :].unsqueeze(dim=1)
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('after_conv_feat', feat_grid, global_step=batch_i)


            # #estimate weights
            if wether_shrink_corr_agg:
                src_weihgts = self.weights_gen(
                    self.L2normalize(F.interpolate(theta, size=(20, 20), mode='bilinear', align_corners=True)))
                tgt_weihgts = self.weights_gen(
                    self.L2normalize(F.interpolate(phi, size=(20, 20), mode='bilinear', align_corners=True)))
                pho_weihgts = self.weights_gen(
                    self.L2normalize(F.interpolate(gama, size=(20, 20), mode='bilinear', align_corners=True)))
            elif corr_agg or flow_agg:
                src_weihgts = self.weights_gen(self.L2normalize(theta))
                tgt_weihgts = self.weights_gen(self.L2normalize(phi))
                pho_weihgts = self.weights_gen(self.L2normalize(gama))


            corr_phi2theta = self.matching_layer(phi_adap,theta_adap) #corr phi2theta  theta_size*phi_size
            # corr_phi2theta = self.L2normalize(corr_phi2theta)

            # corr_phi2theta = self.apply_gaussian_kernel(corr_phi2theta)

            f=corr_phi2theta.reshape(theta_b,theta_h*theta_w,theta_h*theta_w) #reshape and obtain f # theta_size*phi_size   2*(feature_height*feature_width)*(feature_height*feature_width)  phi2theta

            corr_phi2gama = self.matching_layer(phi_adap, gama_adap)  # corr phi2gama
            # corr_phi2gama = self.L2normalize(corr_phi2gama)

            # corr_phi2gama = self.apply_gaussian_kernel(corr_phi2gama)
            f_phi2gama = corr_phi2gama.reshape(theta_b, theta_h * theta_w,
                                       theta_h * theta_w)  # reshape and obtain f # gama*phi_size 2*(feature_height*feature_width)*(feature_height*feature_width)  phi2gama



        find_corr_mode='max' #max or average
        if find_corr_mode=='max':
            # estimate flow
            corr_T2S = f.clone()
            corr_T2S = corr_T2S.reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)  # 2*(s_feature_height*s_feature_width)*t_feature_height*t_feature_width
            # if similarity_computation_mode == 'SANet':
            #     corr_T2S = self.L2normalize(corr_T2S)
            #     corr_T2S = self.apply_gaussian_kernel(corr_T2S)

            #aggregation
            if wether_shrink_corr_agg:
                corr_T2S_temp = F.interpolate(corr_T2S, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(batch_size, 400, theta_h, theta_w)
                corr_T2S_temp = F.interpolate(corr_T2S_temp, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(batch_size, 400, 20, 20)
                corr_T2S_temp = self.agg_corr(corr_T2S_temp, tgt_weihgts)
                corr_T2S_temp = self.agg_corr(corr_T2S_temp, tgt_weihgts)  # aggregation
                corr_T2S_temp = F.interpolate(corr_T2S_temp, size=(theta_h, theta_w), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, 20, 20)
                corr_T2S_temp = F.interpolate(corr_T2S_temp, size=(theta_h, theta_h), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, theta_h, theta_w)
                corr_T2S = torch.cat((corr_T2S, corr_T2S_temp), 1)
                corr_T2S = self.conv_corr(corr_T2S)
            elif corr_agg:
                corr_T2S = self.agg_corr(corr_T2S, tgt_weihgts)  # 1
                # corr_T2S = self.agg_corr(corr_T2S, tgt_weihgts)  # 2
                # corr_T2S = self.agg_corr(corr_T2S, tgt_weihgts)  # 3
                # corr_T2S = self.agg_corr(corr_T2S, tgt_weihgts)  # 4

            if direct_corr2flow:
                flow_T2S=self.conv_corr(corr_T2S)
                grid_T2S, flow_T2S, smoothness_T2S = self.flow2grid_smooth(flow_T2S)
            else:
                grid_T2S, flow_T2S, smoothness_T2S = self.find_correspondence(corr_T2S)


            if flow_agg:
                # aggregation
                flow_T2S = self.agg_flow(flow_T2S, tgt_weihgts)
                grid_T2S, flow_T2S, smoothness_T2S = self.flow2grid_smooth(flow_T2S)

            coor_out['smoothness_T2S'] = smoothness_T2S


            # warping src
            grid_T2S_temp = grid_T2S.permute(0, 3, 1, 2)  # from tgt_image 2  src_image
            grid_T2S_temp = F.interpolate(grid_T2S_temp, size=(image_height, image_width),
                                         mode='bilinear',
                                         align_corners=True)  # b*2*256*256
            grid_T2S_temp = grid_T2S_temp.permute(0, 2, 3, 1)  # b*256*256*2
            warped_src = F.grid_sample(real_img.cuda(), grid_T2S_temp,
                                           mode='bilinear')  # warping the colored original image
            coor_out['warped_src'] = warped_src
            coor_out['real_img'] = real_img


            # estimate flow
            corr_T2P = f_phi2gama.clone()
            corr_T2P = corr_T2P.reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)  # 2*(s_feature_height*s_feature_width)*t_feature_height*t_feature_width

            # ref-attention
            corr_T2P,beta = self.refatten(phi, corr_T2P,pos)
            corr_T2P,beta = self.refatten(phi, corr_T2P,pos)


            # aggregation
            if wether_shrink_corr_agg:
                corr_T2P_temp = F.interpolate(corr_T2P, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(
                                        batch_size, 400, theta_h, theta_w)
                corr_T2P_temp = F.interpolate(corr_T2P_temp, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(
                                        batch_size, 400, 20, 20)
                corr_T2P_temp = self.agg_corr(corr_T2P_temp, tgt_weihgts)  # aggregation
                corr_T2P_temp = F.interpolate(corr_T2P_temp, size=(theta_h, theta_w), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, 20, 20)
                corr_T2P_temp = F.interpolate(corr_T2P_temp, size=(theta_h, theta_h), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, theta_h, theta_w)
                corr_T2P = torch.cat((corr_T2P, corr_T2P_temp), 1)
                corr_T2P = self.conv_corr(corr_T2P)
            elif corr_agg:
                corr_T2P = self.agg_corr(corr_T2P, tgt_weihgts)  # 1
                # corr_T2P = self.agg_corr(corr_T2P, tgt_weihgts)  # 2
                # corr_T2P = self.agg_corr(corr_T2P, tgt_weihgts)  # 3
                # corr_T2P = self.agg_corr(corr_T2P, tgt_weihgts)  # 4


            # if similarity_computation_mode == 'SANet':
            #     corr_T2P = self.L2normalize(corr_T2P)
            #     corr_T2P = self.apply_gaussian_kernel(corr_T2P)
            if direct_corr2flow:
                flow_T2P=self.conv_corr(corr_T2P)
                grid_T2P, flow_T2P, smoothness_T2P = self.flow2grid_smooth(flow_T2P)
            else:
                grid_T2P, flow_T2P, smoothness_T2P = self.find_correspondence(corr_T2P)

            if flow_agg:
                # aggregation
                flow_T2P = self.agg_flow(flow_T2P, tgt_weihgts)
                grid_T2P, flow_T2P, smoothness_T2P = self.flow2grid_smooth(flow_T2P)

            coor_out['smoothness_T2P'] = smoothness_T2P

            # warping photo
            grid_T2P_temp = grid_T2P.permute(0, 3, 1, 2)  # from tgt_image 2  src_image
            grid_T2P_temp = F.interpolate(grid_T2P_temp, size=(image_height, image_width), mode='bilinear',
                                     align_corners=True)  # b*2*256*256
            grid_T2P_temp = grid_T2P_temp.permute(0, 2, 3, 1)  # b*256*256*2
            warped_photo = F.grid_sample(photo.cuda(), grid_T2P_temp,
                                       mode='bilinear')  # warping the colored original image
            coor_out['warped_photo'] = warped_photo
            coor_out['photo'] = photo

            coor_out['flow_T2P'] = flow_T2P  #b*2*64*64
            # print('coor_out[flow_T2P].shape',coor_out['flow_T2P'].shape)
            # os._exit()

            corr_P2T = f_phi2gama.reshape(batch_size, theta_h * theta_w, theta_h * theta_w).permute(0, 2,
                                                                                                  1)  # 2*(t_feature_height*t_feature_width)*(s_feature_height*s_feature_width)
            corr_P2T = corr_P2T.reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)  # 2*(t_feature_height*t_feature_width)*s_feature_height*s_feature_width

            # aggregation
            if wether_shrink_corr_agg:
                corr_P2T_temp = F.interpolate(corr_P2T, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(
                                        batch_size, 400, theta_h, theta_w)
                corr_P2T_temp = F.interpolate(corr_P2T_temp, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(
                                        batch_size, 400, 20, 20)
                corr_P2T_temp = self.agg_corr(corr_P2T_temp, pho_weihgts)  # aggregation
                corr_P2T_temp = F.interpolate(corr_P2T_temp, size=(theta_h, theta_w), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, 20, 20)
                corr_P2T_temp = F.interpolate(corr_P2T_temp, size=(theta_h, theta_h), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, theta_h, theta_w)
                corr_P2T = torch.cat((corr_P2T, corr_P2T_temp), 1)
                corr_P2T = self.conv_corr(corr_P2T)
            elif corr_agg:
                corr_P2T = self.agg_corr(corr_P2T, pho_weihgts)  # 1
                # corr_P2T = self.agg_corr(corr_P2T, pho_weihgts)  # 2
                # corr_P2T = self.agg_corr(corr_P2T, pho_weihgts)  # 3
                # corr_P2T = self.agg_corr(corr_P2T, pho_weihgts)  # 4

            # if similarity_computation_mode == 'SANet':
            #     corr_P2T = self.L2normalize(corr_P2T)
            #     corr_P2T = self.apply_gaussian_kernel(corr_P2T)

            if direct_corr2flow:
                flow_P2T=self.conv_corr(corr_P2T)
                grid_P2T, flow_P2T, smoothness_P2T = self.flow2grid_smooth(flow_P2T)
            else:
                grid_P2T, flow_P2T, smoothness_P2T = self.find_correspondence(corr_P2T)

            if flow_agg:
                # aggregation
                flow_P2T = self.agg_flow(flow_P2T, pho_weihgts)
                grid_P2T, flow_P2T, smoothness_P2T = self.flow2grid_smooth(flow_P2T)

            coor_out['smoothness_P2T'] = smoothness_P2T

            #warp flow
            warped_flow_P2T = -F.grid_sample(flow_P2T, grid_T2P,mode='bilinear')  # warping the flow and negative
            coor_out['warped_flow_P2T'] = warped_flow_P2T



            # estimate flow
            corr_S2T = f.reshape(batch_size, theta_h * theta_w, theta_h * theta_w).permute(0, 2,
                                                                                                  1)  # 2*(t_feature_height*t_feature_width)*(s_feature_height*s_feature_width)
            corr_S2T = corr_S2T.reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)  # 2*(t_feature_height*t_feature_width)*s_feature_height*s_feature_width

            # # ref-attention
            if not self.opt.isTrain:
                feat = corr_S2T.reshape(batch_size, theta_h, theta_w, theta_h,theta_w).permute(0,3,4,1,2).reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)[0, :, :, :].unsqueeze(dim=1)# (s_feature_height*s_feature_width)*1*t_feature_height*t_feature_width
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('score_s2t', feat_grid, global_step=batch_i)

                print('score_s2t',feat)


            corr_S2T,beta = self.refatten(theta, corr_S2T,pos)

            if not self.opt.isTrain:
                feat = beta[0, :, :, :].unsqueeze(dim=1)  # (hq*wq)*1*hk*wk
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('attention', feat_grid, global_step=batch_i)


            corr_S2T,beta = self.refatten(theta, corr_S2T,pos)

            if not self.opt.isTrain:
                feat = beta[0, :, :, :].unsqueeze(dim=1)  # (hq*wq)*1*hk*wk
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('attention_no_pos', feat_grid, global_step=batch_i)

                feat = corr_S2T.reshape(batch_size, theta_h, theta_w, theta_h, theta_w).permute(0, 3, 4, 1, 2).reshape(
                batch_size, theta_h * theta_w, theta_h,
                theta_w)[0, :, :, :].unsqueeze(
                dim=1)  # (s_feature_height*s_feature_width)*1*t_feature_height*t_feature_width
                feat_grid = vutils.make_grid(feat, nrow=6, padding=0, normalize=True, scale_each=True)
                logs_writer.add_image('score_s2t_after_ref', feat_grid, global_step=batch_i)

                print('score_s2t_after_ref', feat)

            if wether_shrink_corr_agg:
                corr_S2T_temp = F.interpolate(corr_S2T, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(
                                        batch_size, 400, theta_h, theta_w)
                corr_S2T_temp = F.interpolate(corr_S2T_temp, size=(20, 20), mode='bilinear', align_corners=True).permute(0, 2, 3,1).reshape(
                                        batch_size, 400, 20, 20)
                corr_S2T_temp = self.agg_corr(corr_S2T_temp, src_weihgts)  # aggregation
                corr_S2T_temp = F.interpolate(corr_S2T_temp, size=(theta_h, theta_w), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, 20, 20)
                corr_S2T_temp = F.interpolate(corr_S2T_temp, size=(theta_h, theta_h), mode='bilinear',
                                              align_corners=True).permute(0, 2, 3, 1).reshape(
                    batch_size, theta_h * theta_w, theta_h, theta_w)
                corr_S2T = torch.cat((corr_S2T, corr_S2T_temp), 1)
                corr_S2T = self.conv_corr(corr_S2T)
            elif corr_agg:
                corr_S2T = self.agg_corr(corr_S2T, src_weihgts)  # 1
                # corr_S2T = self.agg_corr(corr_S2T, src_weihgts)  # 2
                # corr_S2T = self.agg_corr(corr_S2T, src_weihgts)  # 3
                # corr_S2T = self.agg_corr(corr_S2T, src_weihgts)  # 4


            # if similarity_computation_mode == 'SANet':
            #     corr_S2T = self.L2normalize(corr_S2T)
            #     corr_S2T = self.apply_gaussian_kernel(corr_S2T)

            if direct_corr2flow:
                flow_S2T=self.conv_corr(corr_S2T)
                grid_S2T, flow_S2T, smoothness_S2T = self.flow2grid_smooth(flow_S2T)
            else:
                grid_S2T, flow_S2T, smoothness_S2T = self.find_correspondence(corr_S2T)

            if flow_agg:
                # aggregation
                flow_S2T = self.agg_flow(flow_S2T, src_weihgts)
                grid_S2T, flow_S2T, smoothness_S2T = self.flow2grid_smooth(flow_S2T)

            coor_out['smoothness_S2T'] = smoothness_S2T
            coor_out['flow_S2T'] = flow_S2T

            # warp flow
            warped_flow_T2S = -F.grid_sample(flow_T2S, grid_S2T, mode='bilinear')  # warping the flow and negative
            coor_out['warped_flow_T2S'] = warped_flow_T2S


            # warping tgt
            grid_S2T = grid_S2T.permute(0, 3, 1, 2)  # from tgt_image 2  src_image
            grid_S2T = F.interpolate(grid_S2T, size=(image_height, image_width), mode='bilinear',
                                     align_corners=True)  # b*2*256*256
            grid_S2T = grid_S2T.permute(0, 2, 3, 1)  # b*256*256*2
            warped_tgt = F.grid_sample(ref_img.cuda(), grid_S2T,
                                       mode='bilinear')  # warping the colored original image
            coor_out['warped_tgt'] = warped_tgt
            coor_out['ref_img'] = ref_img






        #f_similarity = f.unsqueeze(dim=1)
        # similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        # similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        if WTA_scale_weight == 1:
            f_WTA = f
        else:
            f_WTA = WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        if return_corr:
            return f_WTA
        f_div_C = F.softmax(f_WTA.squeeze(), dim=-1)  # 2*1936*1936; softmax along the horizontal line (dim=-1)


        if find_corr_mode == 'average':
            # estimate flow
            if WTA_scale_weight == 1:
                corr_T2S = f.clone()
            else:
                corr_T2S = WTA_scale.apply(f.clone(), WTA_scale_weight)
            corr_T2S = corr_T2S / temperature

            corr_T2S = F.softmax(corr_T2S.squeeze(), dim=1)  # 2*1936*1936; softmax along the vertical line (dim=1),src
            corr_T2S = corr_T2S.reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)  # 2*(s_feature_height*s_feature_width)*t_feature_height*t_feature_width
            grid_T2S, flow_T2S, smoothness_T2S = self.find_correspondence_average(corr_T2S)

            # warping src
            grid_T2S = grid_T2S.permute(0, 3, 1, 2)  # from tgt_image 2  src_image
            grid_T2S = F.interpolate(grid_T2S, size=(theta_h * self.opt.down, theta_w * self.opt.down), mode='bilinear',
                                     align_corners=True)  # b*2*256*256
            grid_T2S = grid_T2S.permute(0, 2, 3, 1)  # b*256*256*2
            warped_src = F.grid_sample(real_img.cuda(), grid_T2S,
                                       mode='bilinear')  # warping the colored original image
            coor_out['warped_src'] = warped_src
            coor_out['real_img'] = real_img

            if WTA_scale_weight == 1:
                corr_S2T = f.clone()
            else:
                corr_S2T = WTA_scale.apply(f.clone(), WTA_scale_weight)
            corr_S2T = corr_S2T / temperature

            corr_S2T = F.softmax(corr_S2T.squeeze(),
                                 dim=2)  # 2*1936*1936; softmax along the horizontal line (dim=2),tgt
            corr_S2T = corr_S2T.permute(0, 2, 1)  # b*(tgt_h*tgt_w)*(src_h*src_w)

            # corr_S2T = corr_T2S.reshape(batch_size, theta_h * theta_w, theta_h * theta_w).permute(0, 2,
            #                                                                                       1)  # 2*(t_feature_height*t_feature_width)*(s_feature_height*s_feature_width)
            corr_S2T = corr_S2T.reshape(batch_size, theta_h * theta_w, theta_h,
                                        theta_w)  # 2*(t_feature_height*t_feature_width)*s_feature_height*s_feature_width
            grid_S2T, flow_S2T, smoothness_S2T = self.find_correspondence_average(corr_S2T)
            # print('grid_S2T.shape',grid_S2T.shape)
            # print('grid_S2T', grid_S2T)

            # warping tgt
            grid_S2T = grid_S2T.permute(0, 3, 1, 2)  # from tgt_image 2  src_image
            grid_S2T = F.interpolate(grid_S2T, size=(theta_h * self.opt.down, theta_w * self.opt.down), mode='bilinear',
                                     align_corners=True)  # b*2*256*256
            grid_S2T = grid_S2T.permute(0, 2, 3, 1)  # b*256*256*2

            # print('grid_S2T2.shape', grid_S2T.shape)
            # print('grid_S2T2', grid_S2T)

            warped_tgt = F.grid_sample(ref_img.cuda(), grid_S2T,
                                       mode='bilinear')  # warping the colored original image
            coor_out['warped_tgt'] = warped_tgt
            coor_out['ref_img'] = ref_img


            # warped_tgt_center = self.shift_warp(grid_S2T, ref_img, shift_x=0, shift_y=0)
            # coor_out['warped_tgt_center'] = warped_tgt_center.squeeze(2)

            # os._exit()


        # draw matching
        if not self.opt.isTrain:
            # src_points = [
            #     [155.0, 178, 185],
            #     [35.0, 200, 200]]
            src_points_x=torch.tensor(np.linspace(0.2,0.8,5,endpoint=True)*theta_w * self.opt.down)
            src_points_x = src_points_x.unsqueeze(0)
            src_points_x = src_points_x.expand(3, src_points_x.shape[1]).reshape(1, -1)

            src_points_y = torch.tensor(np.linspace(0.4, 0.8, 3, endpoint=True)*theta_h * self.opt.down)
            src_points_y=src_points_y.unsqueeze(1)
            src_points_y = src_points_y.expand(src_points_y.shape[0], 5).reshape(1, -1)

            src_points=torch.cat((src_points_x,src_points_y),0)

            # src_points = torch.as_tensor(src_points)

            grid_np = grid_S2T.cpu().data.numpy()
            est_ref_points = np.zeros((grid_np.shape[0],2, src_points.size(1)))
            # print('est_ref_points.size',est_ref_points.shape)
            for i in range(grid_np.shape[0]):
                for j in range(src_points.size(1)):
                    point_x = int(np.round(src_points[0, j]))
                    point_y = int(np.round(src_points[1, j]))

                    est_y = (grid_np[i, point_y, point_x, 1] + 1) * (theta_h * self.opt.down - 1) / 2
                    est_x = (grid_np[i, point_y, point_x, 0] + 1) * (theta_w * self.opt.down - 1) / 2
                    # print('est_x',est_x)
                    # print('est_y', est_y)
                    est_ref_points[i,:, j] = [est_x, est_y]
                    # print('est_ref_points', est_ref_points)
                    # if i>=2:
                    #     os._exit()


            est_ref_points = torch.as_tensor(est_ref_points)

            # image_syn = torch.cat(
            #     (real_img, ref_img), 3)

            image_syn = torch.cat(
                (input_label.cpu().float(), ref_no_norm), 3)

            btemp = list()
            for i in range(image_syn.shape[0]):
                image_syn_numpy = tensor2arrary_image(image_syn[i,:,:,:].unsqueeze(0).cpu())

                # draw line segment in green
                for j in range(src_points.size(1)):
                    r_rand = float(0.0)
                    g_rand = float(1.0)
                    b_rand = float(0.0)

                    src_point_x = int(np.round(src_points[0, j]))
                    src_point_y = int(np.round(src_points[1, j]))

                    tgt_point_x = int(np.round(est_ref_points[i,0, j]) + theta_w * self.opt.down)
                    tgt_point_y = int(np.round(est_ref_points[i,1, j]))

                    cv2.line(image_syn_numpy, (src_point_x, src_point_y), (tgt_point_x, tgt_point_y),
                             (np.round(255 * b_rand), np.round(255 * g_rand), np.round(255 * r_rand)), 1)
                btemp.append(image_syn_numpy[np.newaxis, :])
            syn_image_numpy = np.concatenate(btemp, axis=0)
            coor_out['syn_image_numpy'] = syn_image_numpy
            # os._exit()





        # downsample the reference color
        # print('ref_img.size',ref_img.size())
        if self.opt.warp_patch:
            ref = F.unfold(ref_img, self.opt.down, stride=self.opt.down)
        else:
            ref = F.avg_pool2d(ref_img, self.opt.down)
            # print('ref.size', ref.size())
            channel = ref.shape[1]
            ref = ref.view(batch_size, channel, -1)
        ref = ref.permute(0, 2, 1)
        # print('ref.size', ref.size())
        # print('f_div_C.size', f_div_C.size())

        y = torch.matmul(f_div_C, ref)  # 2*1936*channel
        # print('y.size', y.size())
        # os.exit()
        if self.opt.warp_patch:
            y = y.permute(0, 2, 1)
            y = F.fold(y, 256, self.opt.down, stride=self.opt.down)
        else:
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*64*64
        if (not self.opt.isTrain) and self.opt.show_corr:
            coor_out['warp_out_bi'] = y if self.opt.warp_patch else self.upsampling_bi(y)
        coor_out['warp_out'] = y if self.opt.warp_patch else self.upsampling(y)

        patch_warping=False
        if patch_warping:
            # #out of memory version
            # channel = ref_img.shape[1]
            # ref = (ref_img.clone()).reshape(batch_size, channel, -1)
            # print('ref_shape',ref.shape)
            # ref = ref.permute(0, 2, 1)#b*(256*256)*3
            #
            # corr_S2T_filter=self.apply_kernel(corr_S2T) #b*(64*64)*64*64
            # corr_S2T_filter = F.interpolate(corr_S2T_filter, size=(theta_h * self.opt.down, theta_w * self.opt.down), mode='bilinear',
            #                          align_corners=True)  # b*(64*64)*256*256
            # corr_S2T_filter=corr_S2T_filter.permute(0,2,3,1).reshape(batch_size,theta_h * self.opt.down*theta_w * self.opt.down,-1)
            # corr_S2T_filter = corr_S2T_filter.reshape(batch_size,theta_h * self.opt.down * theta_w * self.opt.down,theta_h,theta_w)
            # corr_S2T_filter = F.interpolate(corr_S2T_filter, size=(theta_h * self.opt.down, theta_w * self.opt.down),
            #                                 mode='bilinear',
            #                                 align_corners=True)  # b*(256*256)*256*256 #b*(img_h*img_w)*c_h*c_w
            # corr_S2T_filter = corr_S2T_filter.reshape(batch_size,theta_h * self.opt.down * theta_w * self.opt.down,-1) #b*(img_h*img_w)*(c_h*c_w)
            #
            #
            # y = torch.matmul(corr_S2T_filter, ref)  # 2*(256*256)*3
            # coor_out['patch_warp_out'] = y

            # patch warp + upsample
            ref = F.avg_pool2d(ref_img, self.opt.down)
            channel = ref.shape[1]
            ref = ref.view(batch_size, channel, -1)#b*3*(64*64)
            ref = ref.permute(0, 2, 1)#b*(64*64)*3

            corr_S2T_filter = self.apply_kernel(corr_S2T)  # b*(64*64)*64*64 #b*(feat_c_h*feat_c_w)*feat_h*feat_w
            corr_S2T_filter=self.L2normalize(corr_S2T_filter)

            corr_S2T_filter=corr_S2T_filter.reshape(batch_size,theta_h*theta_w,theta_h*theta_w)
            corr_S2T_filter=corr_S2T_filter.permute(0,2,1)#b*(feat_h*feat_w)*(feat_c_h*feat_c_w)#b*4096*4096

            y = torch.matmul(corr_S2T_filter, ref)  # 2*4096*channel
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*64*64

            coor_out['patch_warp_out_upsample'] = self.upsampling(y)

            # #separate warping   out of memory
            # channel = ref_img.shape[1]
            # ref = (ref_img.clone()).reshape(batch_size, channel, -1)
            # print('ref_shape', ref.shape)
            # ref = ref.permute(0, 2, 1)  # b*(256*256)*3
            #
            # warping=[]
            # for i in range(theta_h//self.opt.down):#0-15
            #     warping_horiz=[]
            #     for j in range(theta_w//self.opt.down):#0-15
            #         corr_S2T_small = corr_S2T[:, :, i * self.opt.down:(i + 1) * self.opt.down,
            #                          j * self.opt.down:(j + 1) * self.opt.down]  # b*4096*4*4
            #         print('corr_S2T_small.shape0', corr_S2T_small.shape)  # b*4096*4*4
            #         corr_S2T_small = F.interpolate(corr_S2T_small,size=(int(theta_h / self.opt.down), int(theta_w / self.opt.down )),mode='bilinear',align_corners=True)  # b*4096*16*16
            #         print('corr_S2T_small.shape1', corr_S2T_small.shape)  # b*4096*16*16
            #
            #         corr_S2T_small = corr_S2T_small.permute(0, 2, 3, 1).reshape(batch_size,
            #                                                                     int(theta_h / self.opt.down * theta_w / self.opt.down),
            #                                                                     theta_h, theta_w)
            #         print('corr_S2T_small.shape2', corr_S2T_small.shape)  # b*(16*16)*64*64
            #         corr_S2T_small = F.interpolate(corr_S2T_small,
            #                                        size=(theta_h * self.opt.down, theta_w * self.opt.down),
            #                                        mode='bilinear',
            #                                        align_corners=True)  # b*(16*16)*256*256
            #         print('corr_S2T_small.shape3', corr_S2T_small.shape)  # b*(16*16)*256*256
            #         corr_S2T_small = corr_S2T_small.permute(0, 2, 3, 1).reshape(batch_size,
            #                                                                     theta_h * self.opt.down * theta_w * self.opt.down,
            #                                                                     int(theta_h / self.opt.down),
            #                                                                     int(theta_w / self.opt.down))
            #         print('corr_S2T_small.shape4', corr_S2T_small.shape)  # b*(256*256)*16*16
            #         corr_S2T_small = self.apply_kernel_separate(corr_S2T_small)  # b*(256*256)*16*16
            #         corr_S2T_small = self.L2normalize(corr_S2T_small)
            #         corr_S2T_small = corr_S2T_small.permute(0, 2, 3, 1).reshape(batch_size,
            #                                                                     int(theta_h / self.opt.down * theta_w / self.opt.down),
            #                                                                     -1)
            #         print('corr_S2T_small.shape5', corr_S2T_small.shape)  # b*(16*16)*(256*256)
            #
            #         y = torch.matmul(corr_S2T_small, ref)  # 2*(16*16)*channel
            #         y = y.permute(0, 2, 1).contiguous()
            #         y = y.view(batch_size, channel, int(theta_h / self.opt.down),int(theta_w / self.opt.down))  # 2*3*16*16
            #         print('y.shape',y.shape)
            #
            #         if j==0:
            #             warping_horiz=y
            #         else:
            #             warping_horiz=torch.cat([warping_horiz, y], dim=3)
            #     if i==0:
            #         warping=warping_horiz
            #     else:
            #         warping=torch.cat([warping, warping_horiz], dim=2)
            #
            # coor_out['patch_warp_out'] = warping
            # os.exit()

            ##patch_warp
            ##shift_warp
            # warped_tgt_up=self.shift_warp(grid_S2T, ref_img, shift_x=0, shift_y=-1).unsqueeze(2)
            # warped_tgt_right_up = self.shift_warp(grid_S2T, ref_img, shift_x=1, shift_y=-1).unsqueeze(2)
            # warped_tgt_right = self.shift_warp(grid_S2T, ref_img, shift_x=1, shift_y=0).unsqueeze(2)
            # warped_tgt_right_down = self.shift_warp(grid_S2T, ref_img, shift_x=1, shift_y=1).unsqueeze(2)
            # warped_tgt_down = self.shift_warp(grid_S2T, ref_img, shift_x=0, shift_y=1).unsqueeze(2)
            # warped_tgt_left_down = self.shift_warp(grid_S2T, ref_img, shift_x=-1, shift_y=1).unsqueeze(2)
            # warped_tgt_left = self.shift_warp(grid_S2T, ref_img, shift_x=-1, shift_y=0).unsqueeze(2)
            # warped_tgt_left_up = self.shift_warp(grid_S2T, ref_img, shift_x=-1, shift_y=-1).unsqueeze(2)
            # warped_tgt_center = self.shift_warp(grid_S2T, ref_img, shift_x=0, shift_y=0).unsqueeze(2)
            #
            # warped_tgt_patch=torch.cat((warped_tgt_left_up,warped_tgt_up,warped_tgt_right_up,
            #                             warped_tgt_left,warped_tgt_center,warped_tgt_right,
            #                             warped_tgt_left_down,warped_tgt_down,warped_tgt_right_down),2) #b*3*9*256*256
            #
            # coor_out['warped_tgt_up'] = warped_tgt_up.squeeze(2)
            # coor_out['warped_tgt_center'] = warped_tgt_center.squeeze(2)
            # coor_out['warped_tgt_right_down'] = warped_tgt_right_down.squeeze(2)


            # # shift_confidence
            # print('corr_S2T.shape',corr_S2T.shape)
            # max_index=corr_S2T.max(dim=1)[1]
            # max_idx_y = (max_index // theta_w).view(batch_size, theta_h, theta_w).long()
            # max_idx_x = (max_index % theta_w).view(batch_size, theta_h, theta_w).long()
            # print('max_idx_x.shape',max_idx_x.shape)
            # print('max_idx_x',max_idx_x)
            #
            # # 1-d indices for generating Gaussian kernels
            # idx_x = np.linspace(0, theta_w - 1, theta_w)#64
            # idx_x = torch.tensor(idx_x, dtype=torch.long, requires_grad=False).cuda().unsqueeze(0).unsqueeze(1).expand(batch_size,theta_h,theta_w)#1*1*64  -->   #b*theta_h*theta_w
            # idx_y= np.linspace(0, theta_h - 1, theta_h)
            # idx_y= torch.tensor(idx_y, dtype=torch.long, requires_grad=False).cuda().unsqueeze(0).unsqueeze(2).expand(batch_size,theta_h,theta_w)#1*64*1  -->   #b*theta_h*theta_w
            # print('idx_x.shape', idx_x.shape)
            # print('idx_x',idx_x)
            # print('idx_y', idx_y)


            # max_smlr=self.find_smlr(corr_S2T, max_idx_y, max_idx_x, idx_y, idx_x).unsqueeze(1)
            #
            # print('max_smlr.shape',max_smlr.shape)#b*1*64*64
            # print('max_smlr', max_smlr)
            #
            # max_idx_y_up_left = torch.clamp((max_idx_y-1), 0, theta_h-1, out=None)  # clamp into 0 to 63
            # max_idx_x_up_left = torch.clamp((max_idx_x-1), 0, theta_w-1, out=None)  # clamp into 0 to 63
            # max_smlr_up_left = self.find_smlr(corr_S2T, max_idx_y_up_left, max_idx_x_up_left, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_up = torch.clamp((max_idx_y - 1), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_up = torch.clamp((max_idx_x - 0), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_up = self.find_smlr(corr_S2T, max_idx_y_up, max_idx_x_up, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_up_right = torch.clamp((max_idx_y - 1), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_up_right = torch.clamp((max_idx_x +1), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_up_right = self.find_smlr(corr_S2T, max_idx_y_up_right, max_idx_x_up_right, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_left = torch.clamp((max_idx_y - 0), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_left = torch.clamp((max_idx_x -1), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_left = self.find_smlr(corr_S2T, max_idx_y_left, max_idx_x_left, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_right = torch.clamp((max_idx_y - 0), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_right = torch.clamp((max_idx_x + 1), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_right = self.find_smlr(corr_S2T, max_idx_y_right, max_idx_x_right, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_down_left = torch.clamp((max_idx_y + 1), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_down_left = torch.clamp((max_idx_x - 1), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_down_left = self.find_smlr(corr_S2T, max_idx_y_down_left, max_idx_x_down_left, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_down = torch.clamp((max_idx_y + 1), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_down = torch.clamp((max_idx_x - 0), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_down = self.find_smlr(corr_S2T, max_idx_y_down, max_idx_x_down, idx_y, idx_x).unsqueeze(1)
            #
            # max_idx_y_down_right = torch.clamp((max_idx_y + 1), 0, theta_h - 1, out=None)  # clamp into 0 to 63
            # max_idx_x_down_right = torch.clamp((max_idx_x + 1), 0, theta_w - 1, out=None)  # clamp into 0 to 63
            # max_smlr_down_right = self.find_smlr(corr_S2T, max_idx_y_down_right, max_idx_x_down_right, idx_y, idx_x).unsqueeze(1)
            #
            # max_smlr=torch.cat((max_smlr_up_left,max_smlr_up,max_smlr_up_right,
            #                     max_smlr_left,max_smlr,max_smlr_right,
            #                     max_smlr_down_left,max_smlr_down,max_smlr_down_right
            #                     ),1)
            # print('max_smlr1.shape',max_smlr.shape)#b*9*64*64
            #
            # max_smlr=max_smlr.reshape(batch_size,3,3, theta_h, theta_w).permute(0,3,4,1,2).reshape(batch_size,theta_h*theta_w,3,3)
            #
            # print('max_smlr2.shape', max_smlr.shape)#b*(64*64)*3*3
            #
            # max_smlr = F.interpolate(max_smlr,size=(9,9),mode='bilinear',align_corners=True)  # 3:2gaps->9:8gaps
            # print('max_smlr3.shape', max_smlr.shape)  # b*(64*64)*9*9
            # max_smlr=max_smlr[:,:,3:6,3:6].permute(0,2,3,1).reshape(batch_size,3*3,theta_h,theta_w)
            # print('max_smlr4.shape', max_smlr.shape)  # b*9*64*64
            # max_smlr = F.interpolate(max_smlr,size=(theta_h*self.opt.down,theta_w*self.opt.down),mode='bilinear',align_corners=True)
            # print('max_smlr5.shape', max_smlr.shape)  # b*9*256*256
            #
            # max_smlr=max_smlr.unsqueeze(1).expand_as(warped_tgt_patch)
            # print('max_smlr6.shape', max_smlr.shape)  # b*3*9*256*256
            # max_smlr=self.L2normalize(max_smlr, d=2) #norm to 0-1 and sum to 1
            #
            #
            # warped_tgt_patch=torch.sum(warped_tgt_patch*max_smlr,2)
            # print('warped_tgt_patch.shape',warped_tgt_patch.shape)# b*3*256*256
            #
            # coor_out['patch_warp_out'] = warped_tgt_patch


            # os._exit()

            # patch_warp2 integration
            radius = 4  # Integral multiple of 4
            for shift_y in range(-radius, radius + 1):  # 9*9
                for shift_x in range(-radius, radius + 1):
                    if shift_x == -radius and shift_y == -radius:
                        warped_tgt_patch = self.shift_warp(grid_S2T, ref_img, shift_x=shift_x,
                                                           shift_y=shift_y).unsqueeze(2)
                    else:
                        warped_tgt_patch = torch.cat((warped_tgt_patch,
                                                      self.shift_warp(grid_S2T, ref_img, shift_x=shift_x,
                                                                      shift_y=shift_y).unsqueeze(2)), 2)

            print('warped_tgt_patch.shape', warped_tgt_patch.shape)  # b*3*81*256*256

            # shift_confidence
            print('corr_S2T.shape', corr_S2T.shape)
            max_index = corr_S2T.max(dim=1)[1]
            max_idx_y = (max_index // theta_w).view(batch_size, theta_h, theta_w).long()
            max_idx_x = (max_index % theta_w).view(batch_size, theta_h, theta_w).long()
            print('max_idx_x.shape', max_idx_x.shape)
            print('max_idx_x', max_idx_x)

            # 1-d indices for generating Gaussian kernels
            idx_x = np.linspace(0, theta_w - 1, theta_w)  # 64
            idx_x = torch.tensor(idx_x, dtype=torch.long, requires_grad=False).cuda().unsqueeze(0).unsqueeze(1).expand(
                batch_size, theta_h, theta_w)  # 1*1*64  -->   #b*theta_h*theta_w
            idx_y = np.linspace(0, theta_h - 1, theta_h)
            idx_y = torch.tensor(idx_y, dtype=torch.long, requires_grad=False).cuda().unsqueeze(0).unsqueeze(2).expand(
                batch_size, theta_h, theta_w)  # 1*64*1  -->   #b*theta_h*theta_w
            print('idx_x.shape', idx_x.shape)
            print('idx_x', idx_x)
            print('idx_y', idx_y)

            for shift_y in range(-int(radius/self.opt.down),int(radius/self.opt.down)+1):#9*9--->3*3
                for shift_x in range(-int(radius/self.opt.down),int(radius/self.opt.down)+1):
                    if shift_x==-int(radius/self.opt.down) and shift_y==-int(radius/self.opt.down):
                        max_idx_y = torch.clamp((max_idx_y +shift_y), 0, theta_h - 1, out=None)  # clamp into 0 to 63
                        max_idx_x = torch.clamp((max_idx_x +shift_x), 0, theta_w - 1, out=None)  # clamp into 0 to 63
                        max_smlr = self.find_smlr(corr_S2T, max_idx_y, max_idx_x, idx_y,
                                                          idx_x).unsqueeze(1)
                    else:
                        max_idx_y = torch.clamp((max_idx_y + shift_y), 0, theta_h - 1, out=None)  # clamp into 0 to 63
                        max_idx_x = torch.clamp((max_idx_x + shift_x), 0, theta_w - 1, out=None)  # clamp into 0 to 63
                        max_smlr = torch.cat((max_smlr,self.find_smlr(corr_S2T, max_idx_y, max_idx_x, idx_y,
                                                  idx_x).unsqueeze(1)),1)

            print('max_smlr.shape',max_smlr.shape)#b*9*64*64

            max_smlr = max_smlr.permute(0, 2,3,1).reshape(batch_size,theta_h * theta_w,int(radius/self.opt.down*2+1), int(radius/self.opt.down*2+1))
            print('max_smlr2.shape', max_smlr.shape)  # b*(64*64)*3*3

            max_smlr = F.interpolate(max_smlr, size=(radius*2+1, radius*2+1), mode='bilinear', align_corners=True)  # 3:2gaps->9:8gaps
            print('max_smlr3.shape', max_smlr.shape)  # b*(64*64)*9*9
            max_smlr = max_smlr.permute(0, 2, 3, 1).reshape(batch_size, (radius*2+1) * (radius*2+1), theta_h, theta_w)
            print('max_smlr4.shape', max_smlr.shape)  # b*81*64*64
            max_smlr = F.interpolate(max_smlr, size=(theta_h * self.opt.down, theta_w * self.opt.down), mode='bilinear',
                                     align_corners=True)
            print('max_smlr5.shape', max_smlr.shape)  # b*81*256*256

            max_smlr = max_smlr.unsqueeze(1).expand_as(warped_tgt_patch)
            print('max_smlr6.shape', max_smlr.shape)  # b*3*81*256*256
            max_smlr = self.L2normalize(max_smlr, d=2)  # norm to 0-1 and sum to 1

            warped_tgt_patch = torch.sum(warped_tgt_patch * max_smlr, 2)
            print('warped_tgt_patch.shape', warped_tgt_patch.shape)  # b*3*256*256

            coor_out['patch_warp_out'] = warped_tgt_patch

            # os._exit()




        # if find_corr_mode == 'max':
        #     ref_seg = F.interpolate(ref_seg_map, scale_factor=1 , mode='nearest')# b*512*64*64
        #     # warping ref_seg
        #     grid_S2T = grid_S2T.permute(0, 3, 1, 2)  # from src_image 2  tgt_image
        #     grid_S2T = F.interpolate(grid_S2T, scale_factor=1/self.down , mode='bilinear',
        #                              align_corners=True)  # b*2*64*64
        #     grid_S2T = grid_S2T.permute(0, 2, 3, 1)  # b*64*64*2
        #     warp_mask = F.grid_sample(ref_seg.cuda(), grid_S2T,
        #                                mode='bilinear')  # warping the colored original image
        #     coor_out['warp_mask'] = warp_mask



            # os._exit()
        elif self.opt.warp_mask_losstype == 'cycle':
            f_div_C_v = F.softmax(f_WTA.transpose(1, 2), dim=-1)  # 2*1936*1936; softmax along the vertical line
            seg = F.interpolate(seg_map, scale_factor=1 / self.opt.down, mode='nearest')
            channel = seg.shape[1]
            seg = seg.view(batch_size, channel, -1)
            seg = seg.permute(0, 2, 1)
            warp_mask_to_ref = torch.matmul(f_div_C_v, seg)  # 2*1936*channel
            warp_mask = torch.matmul(f_div_C, warp_mask_to_ref)  # 2*1936*channel
            warp_mask = warp_mask.permute(0, 2, 1).contiguous()
            coor_out['warp_mask'] = warp_mask.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        else:
            warp_mask = None

        if self.opt.warp_cycle_w > 0:
            f_div_C_v = F.softmax(f_WTA.transpose(1, 2), dim=-1)
            if self.opt.warp_patch:
                y = F.unfold(y, self.opt.down, stride=self.opt.down)
                y = y.permute(0, 2, 1)
                warp_cycle = torch.matmul(f_div_C_v, y)
                warp_cycle = warp_cycle.permute(0, 2, 1)
                warp_cycle = F.fold(warp_cycle, 256, self.opt.down, stride=self.opt.down)
                coor_out['warp_cycle'] = warp_cycle
            else:
                channel = y.shape[1]
                y = y.view(batch_size, channel, -1).permute(0, 2, 1)
                warp_cycle = torch.matmul(f_div_C_v, y).permute(0, 2, 1).contiguous()
                coor_out['warp_cycle'] = warp_cycle.view(batch_size, channel, feature_height, feature_width)
                if self.opt.two_cycle:
                    real_img = F.avg_pool2d(real_img, self.opt.down)
                    real_img = real_img.view(batch_size, channel, -1)
                    real_img = real_img.permute(0, 2, 1)
                    warp_i2r = torch.matmul(f_div_C_v, real_img).permute(0, 2, 1).contiguous()  #warp input to ref
                    warp_i2r = warp_i2r.view(batch_size, channel, feature_height, feature_width)
                    warp_i2r2i = torch.matmul(f_div_C, warp_i2r.view(batch_size, channel, -1).permute(0, 2, 1))
                    coor_out['warp_i2r'] = warp_i2r
                    coor_out['warp_i2r2i'] = warp_i2r2i.permute(0, 2, 1).contiguous().view(batch_size, channel, feature_height, feature_width)

        return coor_out
