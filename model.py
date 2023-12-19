import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import time

from models.position_encoding import build_position_encoding
from models.token_transformer import Token_transformer
from models.token_transformer import Token_transformer_interaction

from torch.autograd import Variable
from gluoncv import gluoncvth
# import gluoncvth as gcv


# some parts of codes are from 'https://github.com/ignacio-rocco/weakalign'

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            # self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            # self.attention1 = Token_transformer(dim=in_chans * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention1 = Token_transformer(dim=in_chans * 5*5, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0)
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

class T2T_module_interaction(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='transformer', in_chans=256, embed_dim=400, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            # self.soft_split0 = nn.Unfold(kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            # self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            # self.attention1 = Token_transformer(dim=in_chans * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention1 = Token_transformer_interaction(dim=in_chans, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer_interaction(dim=in_chans, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0)
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

    def forward(self, x,y):
        # step0: soft split
        # x = self.soft_split0(x).transpose(1, 2)

        # iteration1: restricturization/reconstruction
        y = self.attention1(x,y)
        # B, new_HW, C = x.shape
        # x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # # iteration1: soft split
        # x = self.soft_split1(x).transpose(1, 2)
        #
        # iteration2: restricturization/reconstruction
        y = self.attention2(x,y)
        # x = self.attention2(x)
        # B, new_HW, C = x.shape
        # x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # # iteration2: soft split
        # x = self.soft_split2(x).transpose(1, 2)
        #
        # # final tokens
        # x = self.project(x)

        # x = self.project3_2(self.project3_1(x))


        # #reshape
        # B, new_HW, C = x.shape
        # x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        return y


class T2T_module4(nn.Module):
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
            self.attention1 = Token_transformer(dim=in_chans * 3*3, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0)
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


class T2T_module2(nn.Module):
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
            self.attention1 = Token_transformer(dim=in_chans * 3*3, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0)
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
        self.beta_gamma = nn.Parameter(torch.tensor(10.), requires_grad=True)
        # self.gamma = Variable(torch.tensor(0.0), requires_grad=True)
        #
        self.sum_gamma0 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.sum_gamma1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        # self.sum_gamma0 = Variable(torch.tensor(0.0), requires_grad=True)
        # self.sum_gamma1 = Variable(torch.tensor(0.0), requires_grad=True)

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
        # q = q * scaling

        q = self.L2normalize(q)
        k = self.L2normalize(k)

        q = q.view(-1, self.cout, x.shape[2] * x.shape[3])#b*c*(hq*wq)
        k = k.view(-1, self.cout, x.shape[2] * x.shape[3])#b*c*(hk*wk)
        v = v.view(-1, v.shape[1], v.shape[2] * v.shape[3])#b*cv*(hv*wv)


        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(q.permute(0, 2, 1), k)*self.beta_gamma, -1)  # b*(hq*wq)*(hk*wk)
        # print('beta', beta)
        # print('torch.max(beta,2)', torch.max(beta, 2))
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
        # print('self.gamma', self.gamma)
        # print('self.beta_gamma', self.beta_gamma)
        # print('torch.exp(self.sum_gamma0)/gamma_total', torch.exp(self.sum_gamma0)/gamma_total)
        # print('torch.exp(self.sum_gamma1)/gamma_total', torch.exp(self.sum_gamma1) / gamma_total)



        # return out.permute(0,3,1,2)
        return  out,beta.reshape(-1, beta.shape[1], x.shape[2], x.shape[3])
        # return o

# class FeatureExtraction(nn.Module):
#     def __init__(self):
#         super(FeatureExtraction, self).__init__()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.model = models.resnet101(pretrained=True)
#         self.model = gluoncvth.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
#         # print('self.model',self.model)
#         resnet_feature_layers = ['conv1',
#                                  'bn1',
#                                  'relu',
#                                  'maxpool',
#                                  'layer1',
#                                  'layer2',
#                                  'layer3',
#                                  'layer4',]
#         layer1 = 'layer1'
#         layer2 = 'layer2'
#         layer3 = 'layer3'
#         layer4 = 'layer4'
#         layer1_idx = resnet_feature_layers.index(layer1)
#         layer2_idx = resnet_feature_layers.index(layer2)
#         layer3_idx = resnet_feature_layers.index(layer3)
#         layer4_idx = resnet_feature_layers.index(layer4)
#         resnet_module_list = [self.model.conv1,
#                               self.model.bn1,
#                               self.model.relu,
#                               self.model.maxpool,
#                               self.model.layer1,
#                               self.model.layer2,
#                               self.model.layer3,
#                               self.model.layer4]
#         self.layer1 = nn.Sequential(*resnet_module_list[:layer1_idx + 1])
#         self.layer2 = nn.Sequential(*resnet_module_list[layer1_idx + 1:layer2_idx + 1])
#         self.layer3 = nn.Sequential(*resnet_module_list[layer2_idx + 1:layer3_idx + 1])
#         self.layer4 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])

#         # print('self.layer1', self.layer1)
#         # print('self.layer2', self.layer2)
#         # print('self.layer3', self.layer3)
#         # print('self.layer4', self.layer4)

#         for param in self.layer1.parameters():
#             param.requires_grad = False
#         for param in self.layer2.parameters():
#             param.requires_grad = False
#         for param in self.layer3.parameters():
#             param.requires_grad = False
#         for param in self.layer4.parameters():
#             param.requires_grad = False

#     def forward(self, image_batch):
#         layer1_feat = self.layer1(image_batch)
#         layer2_feat = self.layer2(layer1_feat)
#         layer3_feat = self.layer3(layer2_feat)
#         layer4_feat = self.layer4(layer3_feat)
#         return layer1_feat, layer2_feat, layer3_feat, layer4_feat

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = models.resnet101(pretrained=True)
        self.model = gluoncvth.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
        # print('self.model',self.model)
        # resnet_feature_layers = ['conv1',
        #                          'bn1',
        #                          'relu',
        #                          'maxpool',
        #                          'layer1',
        #                          'layer2',
        #                          'layer3',
        #                          'layer4',]
        # layer1 = 'layer1'
        # layer2 = 'layer2'
        # layer3 = 'layer3'
        # layer4 = 'layer4'
        # layer1_idx = resnet_feature_layers.index(layer1)
        # layer2_idx = resnet_feature_layers.index(layer2)
        # layer3_idx = resnet_feature_layers.index(layer3)
        # layer4_idx = resnet_feature_layers.index(layer4)
        # resnet_module_list = [self.model.conv1,
        #                       self.model.bn1,
        #                       self.model.relu,
        #                       self.model.maxpool,
        #                       self.model.layer1,
        #                       self.model.layer2,
        #                       self.model.layer3,
        #                       self.model.layer4]
        # self.layer1 = nn.Sequential(*resnet_module_list[:layer1_idx + 1])
        # self.layer2 = nn.Sequential(*resnet_module_list[layer1_idx + 1:layer2_idx + 1])
        # self.layer3 = nn.Sequential(*resnet_module_list[layer2_idx + 1:layer3_idx + 1])
        # self.layer4 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])
        resnet_feature_layers = ['conv1',
                                 'bn1',
                                 'relu',
                                 'maxpool',
                                 'layer1',
                                 'layer2',
                                 'layer3',
                                 'layer4',
                                 'avgpool',
                                 'fc',]
        layer1 = 'layer1'
        layer2 = 'layer2'
        layer3 = 'layer3'
        layer4 = 'layer4'
        layer1_idx = resnet_feature_layers.index(layer1)
        layer2_idx = resnet_feature_layers.index(layer2)
        layer3_idx = resnet_feature_layers.index(layer3)
        layer4_idx = resnet_feature_layers.index(layer4)
        resnet_module_list = [self.model.conv1,
                              self.model.bn1,
                              self.model.relu,
                              self.model.maxpool,
                              self.model.layer1,
                              self.model.layer2,
                              self.model.layer3,
                              self.model.layer4,
                              self.model.avgpool,
                              self.model.fc,]
        self.layer1 = nn.Sequential(*resnet_module_list[:layer1_idx + 1])
        self.layer2 = nn.Sequential(*resnet_module_list[layer1_idx + 1:layer2_idx + 1])
        self.layer3 = nn.Sequential(*resnet_module_list[layer2_idx + 1:layer3_idx + 1])
        self.layer4 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])
        self.rest = nn.Sequential(*resnet_module_list[layer4_idx + 1:])

        # print('self.layer1', self.layer1)
        # print('self.layer2', self.layer2)
        # print('self.layer3', self.layer3)
        # print('self.layer4', self.layer4)
        # print('self.rest', self.rest)


        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        for param in self.rest.parameters():
            param.requires_grad = False

    def forward(self, image_batch):
        layer1_feat = self.layer1(image_batch)
        layer2_feat = self.layer2(layer1_feat)
        layer3_feat = self.layer3(layer2_feat)
        layer4_feat = self.layer4(layer3_feat)
        return layer1_feat, layer2_feat, layer3_feat, layer4_feat


class adap_layer_feat3(nn.Module):
    def __init__(self):
        super(adap_layer_feat3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )


    def forward(self, feature):
        """
        We empirically found that applying ReLU before addition is better than applying ReLU after addition,
        but it might not be the optimal architecture.
        We've focussed more on establishing correspondences
        and we will make adaptation layers more efficient in the near future.
        """
        feature = feature + self.conv1(feature)
        feature = feature + self.conv2(feature)
        # print('self.conv1.parameters in token printed start')
        # for para in self.conv1.parameters():
        #     print(para)
        # print('self.conv1.parameters in token printed over')
        # for para in self.conv2.parameters():
        #     print(para)
        # print('self.conv2.parameters in token printed over')

        return feature
    
class adap_layer_feat4(nn.Module):
    def __init__(self):
        super(adap_layer_feat4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
            
    def forward(self, feature):
        feature = feature + self.conv1(feature)
        feature = feature + self.conv2(feature)
        return feature
    
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
        corr = corr.view(b, h2 * w2, h1, w1) # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr


class weights_gen(nn.Module):
    def __init__(self):
        super(weights_gen, self).__init__()
        self.conv1=nn.Conv2d(1024, 9, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1)
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
        weights=weights.reshape(-1, 9, h*w)
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
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)
            volume_output_revert[:, :, (w + shift_x):w, :, :] = volume_output_revert[:, :,
                                                                (w + shift_x - 1):(w + shift_x), :,
                                                                :]  # right margin padding
            volume_second = volume_output_revert.reshape(1, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # left shift for computing volume aggregation
            volume[:, :, :, (w + shift_x):w] = volume[:, :, :,
                                               (w + shift_x - 1):(w + shift_x)]  # right margin padding for volume
        if shift_x > 0:  # right shift,leftvolume
            theta = torch.tensor([[1, 0, (2.0 / (w - 1)) * (-shift_x)], [0, 1, 0]],
                                 dtype=torch.float32)  # x= x - 2.0 / 19  right shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)
            volume_output_revert[:, :, 0:shift_x, :, :] = volume_output_revert[:, :, shift_x:(shift_x + 1), :,
                                                          :]  # left margin padding
            volume_second = volume_output_revert.reshape(1, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # right shift for computing volume aggregation
            volume[:, :, :, 0:shift_x] = volume[:, :, :, shift_x:(shift_x + 1)]  # left margin padding for volume
        if shift_y > 0:  # down shift,upvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h - 1)) * (-shift_y)]],
                                 dtype=torch.float32)  # y= y - 2.0 / 19  down shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)  # b * h * w * h * w
            volume_output_revert[:, 0:shift_y, :, :, :] = volume_output_revert[:, shift_y:(shift_y + 1), :, :,
                                                          :]  # up margin padding
            volume_second = volume_output_revert.reshape(1, h * w, h, w)
            volume = F.grid_sample(volume_second, grid)  # down shift for computing volume aggregation
            volume[:, :, 0:shift_y, :] = volume[:, :, shift_y:(shift_y + 1), :]  # up margin padding for volume
        if shift_y < 0:  # up shift,downvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h - 1)) * (-shift_y)]],
                                 dtype=torch.float32)  # y= y + 2.0 / 19  up shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h * w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)  # b * h' * w' * h * w
            volume_output_revert[:, (w + shift_y):w, :, :, :] = volume_output_revert[:, (w + shift_y - 1):(w + shift_y),
                                                                :, :, :]  # down margin padding
            volume_second = volume_output_revert.reshape(1, h * w, h, w)
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

class aggregation(nn.Module):
    def __init__(self):
        super(aggregation, self).__init__()
        self.conv1 = nn.Conv3d(5, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0),bias=False)
        # x=torch.tensor([[[[[0.0]]],[[[0.0]]],[[[1.0]]],[[[0.0]]],[[[0.0]]]]])
        # self.conv1.weight=torch.nn.Parameter(x)
        self.relu = nn.ReLU()

    def shiftVol(self, volume,shift_x=0,shift_y=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b, c, h, w = volume.size()
        if shift_x<0:#left shift,rightvolume
            theta = torch.tensor([[1, 0, (2.0 / (w-1))*(-shift_x)], [0, 1, 0]], dtype=torch.float32) # x= x + 2.0 / 19  left shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h*w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)
            volume_output_revert[:, :, (w+shift_x):w, :, :] = volume_output_revert[:, :, (w+shift_x-1):(w+shift_x), :, :]  # right margin padding
            volume_second = volume_output_revert.reshape(1, h*w, h, w)
            volume = F.grid_sample(volume_second, grid)  # left shift for computing volume aggregation
            volume[:, :, :, (w+shift_x):w] = volume[:, :, :, (w+shift_x-1):(w+shift_x)]  # right margin padding for volume
        if shift_x>0:#right shift,leftvolume
            theta = torch.tensor([[1, 0, (2.0 / (w-1))*(-shift_x)], [0, 1, 0]], dtype=torch.float32)  # x= x - 2.0 / 19  right shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h*w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)
            volume_output_revert[:, :, 0:shift_x, :, :] = volume_output_revert[:, :, shift_x:(shift_x+1), :, :]  # left margin padding
            volume_second = volume_output_revert.reshape(1, h*w, h, w)
            volume = F.grid_sample(volume_second, grid)  # right shift for computing volume aggregation
            volume[:, :, :, 0:shift_x] = volume[:, :, :, shift_x:(shift_x+1)]  # left margin padding for volume
        if shift_y > 0:  # down shift,upvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h-1))*(-shift_y)]], dtype=torch.float32)  # y= y - 2.0 / 19  down shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h*w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)  # b * h * w * h * w
            volume_output_revert[:, 0:shift_y, :, :, :] = volume_output_revert[:, shift_y:(shift_y+1), :, :, :]  # up margin padding
            volume_second = volume_output_revert.reshape(1, h*w, h, w)
            volume = F.grid_sample(volume_second, grid)  # down shift for computing volume aggregation
            volume[:, :, 0:shift_y, :] = volume[:, :, shift_y:(shift_y+1), :]  # up margin padding for volume
        if shift_y < 0:  # up shift,downvolume
            theta = torch.tensor([[1, 0, 0], [0, 1, (2.0 / (h-1))*(-shift_y)]], dtype=torch.float32)  # y= y + 2.0 / 19  up shift
            volume_temp = volume.reshape(1, h, w, h, w).permute(0, 3, 4, 1, 2).reshape(1, h*w, h, w)
            grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
            grid = grid.to(device)
            volume_output = F.grid_sample(volume_temp, grid)
            volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, h, w, h, w)  # b * h' * w' * h * w
            volume_output_revert[:, (w+shift_y):w, :, :, :] = volume_output_revert[:, (w+shift_y-1):(w+shift_y), :, :, :]  # down margin padding
            volume_second = volume_output_revert.reshape(1, h*w, h, w)
            volume = F.grid_sample(volume_second, grid)  # up shift for computing volume aggregation
            volume[:, :, (w+shift_y):w, :] = volume[:, :, (w+shift_y-1):(w+shift_y), :]  # down margin padding for volume
        return volume

    def forward(self, corr):
        #corr_S2T_right = self.rightvolume(corr)
        b, c, h, w = corr.size()
        corr_S2T_right = self.shiftVol(corr,shift_x=-1,shift_y=0)
        corr_S2T_right=corr_S2T_right.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w) #b * c *  (h * w) * h' * w'
        #corr_S2T_left = self.leftvolume(corr)
        corr_S2T_left = self.shiftVol(corr,shift_x=1,shift_y=0)
        corr_S2T_left = corr_S2T_left.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        #corr_S2T_up = self.upvolume(corr)
        corr_S2T_up = self.shiftVol(corr,shift_x=0,shift_y=1)
        corr_S2T_up = corr_S2T_up.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        #corr_S2T_down = self.downvolume(corr)
        corr_S2T_down = self.shiftVol(corr,shift_x=0,shift_y=-1)
        corr_S2T_down = corr_S2T_down.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'
        corr=corr.unsqueeze(1).reshape(-1, 1, h, w, h, w).permute(0, 1, 4, 5, 2, 3).reshape(
            -1, 1, (h * w), h, w)  # b * c *  (h * w) * h' * w'     
        #5layers for each pixel
        corr_cat=torch.cat((corr_S2T_up,corr_S2T_right,corr,corr_S2T_down,corr_S2T_left),1)# b * 4 *  (h * w) * h' * w'
        corr= self.relu(self.conv1(corr_cat))# b * 1 *  (h * w) * h' * w'
        corr=corr.squeeze(1).reshape(-1, h, w, h, w).permute(0, 3,4,1,2).reshape(
            -1, (h * w), h, w)  # b * (h' * w') * h * w
        
        return corr


class apply_gaussian_kernel(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(apply_gaussian_kernel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma

        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
                                               np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).to(device)
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).to(device)

        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,
                                                                                                                  1,
                                                                                                                  3).to(
            device)
        self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
                                                                                                                  3,
                                                                                                                  1).to(
            device)

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(device)
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(device)

        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1, 1, feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).to(device)
        self.y_normal = np.linspace(-1, 1, feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).to(device)

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

class find_correspondence(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(find_correspondence, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma
        
        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1,1,feature_W), np.linspace(-1,1,feature_H)) # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).to(device)
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).to(device)
        
        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1,0,1], dtype=torch.float, requires_grad=False).view(1,1,1,3).expand(1,2,1,3).to(device)
        self.dy_kernel = torch.tensor([-1,0,1], dtype=torch.float, requires_grad=False).view(1,1,3,1).expand(1,2,3,1).to(device)
        
        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0,feature_W-1,feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(device)
        self.y = np.linspace(0,feature_H-1,feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(device)
        
        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1,1,feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).to(device)
        self.y_normal = np.linspace(-1,1,feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).to(device)

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1] # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()
        
        x = self.x.view(1,1,w,1,1).expand(b, 1, w, h, w)
        y = self.y.view(1,h,1,1,1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr
    
    def softmax_with_temperature(self, x, beta, d = 1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(beta*x)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def rightvolume(self, volume):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        theta = torch.tensor([[1, 0, 2.0 / 19], [0, 1, 0]], dtype=torch.float32) # x= x + 2.0 / 19  left shift
        volume_temp = volume.reshape(1, 20, 20, 20, 20).permute(0, 3, 4, 1, 2).reshape(1, 400, 20, 20)
        grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
        grid=grid.to(device)
        volume_output = F.grid_sample(volume_temp, grid)
        volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, 20, 20, 20, 20)
        volume_output_revert[:, :, 19, :, :] = volume_output_revert[:, :, 18, :, :]  # right margin padding
        volume_second = volume_output_revert.reshape(1, 400, 20, 20)
        volume_second_output = F.grid_sample(volume_second, grid) # left shift for computing volume aggregation
        volume_second_output[:, :, :, 19] = volume_second_output[:, :, :, 18] # right margin padding for volume
        return volume_second_output

    def leftvolume(self, volume):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        theta = torch.tensor([[1, 0, -2.0 / 19], [0, 1, 0]], dtype=torch.float32)  # x= x - 2.0 / 19  right shift
        volume_temp = volume.reshape(1, 20, 20, 20, 20).permute(0, 3, 4, 1, 2).reshape(1, 400, 20, 20)
        grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
        grid=grid.to(device)
        volume_output = F.grid_sample(volume_temp, grid)
        volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, 20, 20, 20, 20)
        volume_output_revert[:, :, 0, :, :] = volume_output_revert[:, :, 1, :, :]  # left margin padding
        volume_second = volume_output_revert.reshape(1, 400, 20, 20)
        volume_second_output = F.grid_sample(volume_second, grid)  # right shift for computing volume aggregation
        volume_second_output[:, :, :, 0] = volume_second_output[:, :, :, 1]  # left margin padding for volume
        return volume_second_output

    def upvolume(self, volume):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        theta = torch.tensor([[1, 0, 0], [0, 1, -2.0 / 19]], dtype=torch.float32)  # y= y - 2.0 / 19  down shift
        volume_temp = volume.reshape(1, 20, 20, 20, 20).permute(0, 3, 4, 1, 2).reshape(1, 400, 20, 20)
        grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
        grid=grid.to(device)
        volume_output = F.grid_sample(volume_temp, grid)
        volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, 20, 20, 20, 20) # b * h * w * h * w
        volume_output_revert[:, 0, :, :, :] = volume_output_revert[:, 1, :, :, :]  # up margin padding
        volume_second = volume_output_revert.reshape(1, 400, 20, 20)
        volume_second_output = F.grid_sample(volume_second, grid)  # down shift for computing volume aggregation
        volume_second_output[:, :, 0, :] = volume_second_output[:, :, 1, :]  # up margin padding for volume
        return volume_second_output

    def downvolume(self, volume):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        theta = torch.tensor([[1, 0, 0], [0, 1, 2.0 / 19]], dtype=torch.float32)  # y= y + 2.0 / 19  up shift
        volume_temp = volume.reshape(1, 20, 20, 20, 20).permute(0, 3, 4, 1, 2).reshape(1, 400, 20, 20)
        grid = F.affine_grid(theta.unsqueeze(0), volume_temp.size())
        grid=grid.to(device)
        volume_output = F.grid_sample(volume_temp, grid)
        volume_output_revert = volume_output.permute(0, 2, 3, 1).reshape(1, 20, 20, 20, 20) # b * h * w * h * w
        volume_output_revert[:, 19, :, :, :] = volume_output_revert[:, 18, :, :, :]  # down margin padding
        volume_second = volume_output_revert.reshape(1, 400, 20, 20)
        volume_second_output = F.grid_sample(volume_second, grid)  # up shift for computing volume aggregation
        volume_second_output[:, :, 19, :] = volume_second_output[:, :, 18, :]  # down margin padding for volume
        return volume_second_output

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)
    
    def kernel_soft_argmax(self, corr):
        b,_,h,w = corr.size()

        corr = self.softmax_with_temperature(corr, beta=self.beta, d=1)

        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def get_flow_smoothness(self, flow, GT_mask):
        flow_dx = F.conv2d(F.pad(flow,(1,1,0,0)),self.dx_kernel)/2 # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow,(0,0,1,1)),self.dy_kernel)/2 # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx) * GT_mask # consider foreground regions only
        flow_dy = torch.abs(flow_dy) * GT_mask
        
        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def get_flow_smoothness_nomask(self, flow):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx)   # consider whole regions
        flow_dy = torch.abs(flow_dy)

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness
    
    def forward(self, corr, GT_mask = None):
        b,_,h,w = corr.size()
        grid_X = self.grid_X.expand(b, h, w) # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1) # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w) # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)
                
        if self.beta is not None:
            grid_x, grid_y = self.kernel_soft_argmax(corr)
        else: # discrete argmax
            _,idx = torch.max(corr,dim=1)
            grid_x = idx % w
            grid_x = (grid_x.float() / (w-1) - 0.5) * 2
            grid_y = idx // w
            grid_y = (grid_y.float() / (h-1) - 0.5) * 2
        
        grid = torch.cat((grid_x.permute(0,2,3,1), grid_y.permute(0,2,3,1)),3) # 2-channels@3rd-dim, first channel for x / second channel for y
        flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),1) # 2-channels@1st-dim, first channel for x / second channel for y
        
        if GT_mask is None: # test
            smoothness = self.get_flow_smoothness_nomask(flow)
            return grid, flow, smoothness
        else: # train
            smoothness = self.get_flow_smoothness(flow,GT_mask)
            return grid, flow, smoothness

class SFNet(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma,istrain=False):
        super(SFNet, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extraction = FeatureExtraction()
        # self.adap_layer_feat3 = adap_layer_feat3()
        # self.adap_layer_feat4 = adap_layer_feat4()
        # self.weights_gen=weights_gen()
        self.matching_layer = matching_layer()
        # self.aggregation = aggregation()
        self.apply_gaussian_kernel = apply_gaussian_kernel(feature_H, feature_W, beta, kernel_sigma)
        # self.agg_corr = agg_corr()
        self.find_correspondence = find_correspondence(feature_H, feature_W, beta, kernel_sigma)

        # chans_num_input=1024
        # chans_num_token=1024
        # chans_num_ouput = 1024

        # self.tokens_to_token3_1 = T2T_module(tokens_type='transformer', in_chans=chans_num_input, embed_dim=chans_num_token)
        # self.tokens_to_token3_2 = T2T_module(tokens_type='transformer', in_chans=chans_num_token,
        #                                embed_dim=chans_num_ouput)

        # self.position_embedding3 = build_position_encoding(chans_num_input, scale=5*5)
        # # self.position_embedding2 = build_position_encoding(chans_num_input)
        # self.instnorm3_1 = nn.Sequential(
        #     nn.InstanceNorm2d(chans_num_token),
        #     nn.ReLU()
        # )
        # self.instnorm3_2 = nn.Sequential(
        #     nn.InstanceNorm2d(chans_num_ouput),
        #     nn.ReLU()
        # )

        # self.refatten = refattention(chans_num_input, chans_num_ouput, feature_H, feature_W)

        chans_num_input = 1024*2
        chans_num_token = 1024*2
        chans_num_ouput = 1024*2
        # self.tokens_to_token4_1 = T2T_module4(tokens_type='transformer', in_chans=chans_num_input,
        #                                   embed_dim=chans_num_token)
        # self.tokens_to_token4_2 = T2T_module4(tokens_type='transformer', in_chans=chans_num_token,
        #                                    embed_dim=chans_num_ouput)
        #
        # self.position_embedding4 = build_position_encoding(chans_num_input, scale=3*3)
        self.tokens_to_token4_1 = T2T_module(tokens_type='transformer', in_chans=chans_num_input,
                                              embed_dim=chans_num_token)
        self.tokens_to_token4_2 = T2T_module(tokens_type='transformer', in_chans=chans_num_token,
                                              embed_dim=chans_num_ouput)

        self.position_embedding4 = build_position_encoding(chans_num_input, scale=5*5)
        self.instnorm4_1 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_token),
            nn.ReLU()
        )
        self.instnorm4_2 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_ouput),
            nn.ReLU()
        )

        self.conv_embed_4 = nn.Sequential(
            nn.Conv2d(chans_num_ouput*2, 256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.feat_merge_4 = nn.Sequential(
            nn.Conv2d(chans_num_ouput, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        chans_num_input=1024
        chans_num_token=1024
        chans_num_ouput = 1024

        self.tokens_to_token3_1 = T2T_module(tokens_type='transformer', in_chans=chans_num_input, embed_dim=chans_num_token)
        self.tokens_to_token3_2 = T2T_module(tokens_type='transformer', in_chans=chans_num_token,
                                       embed_dim=chans_num_ouput)

        self.position_embedding3 = build_position_encoding(chans_num_input, scale=5*5)
        # self.position_embedding2 = build_position_encoding(chans_num_input)
        self.instnorm3_1 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_token),
            nn.ReLU()
        )
        self.instnorm3_2 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_ouput),
            nn.ReLU()
        )

        self.conv_embed_3 = nn.Sequential(
            nn.Conv2d(chans_num_ouput*2, 256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.feat_merge_3 = nn.Sequential(
            nn.Conv2d(chans_num_ouput, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )



        chans_num_input = int(1024 / 2)
        chans_num_token = int(1024 / 2)
        chans_num_ouput = int(1024 / 2)
        self.tokens_to_token2_1 = T2T_module(tokens_type='transformer', in_chans=chans_num_input,
                                              embed_dim=chans_num_token)
        self.tokens_to_token2_2 = T2T_module(tokens_type='transformer', in_chans=chans_num_token,
                                              embed_dim=chans_num_ouput)

        self.position_embedding2 = build_position_encoding(chans_num_input, scale=5*5)
        self.instnorm2_1 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_token),
            nn.ReLU()
        )
        self.instnorm2_2 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_ouput),
            nn.ReLU()
        )

        self.conv_embed_2 = nn.Sequential(
            nn.Conv2d(chans_num_ouput*2, 256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.feat_merge_2 = nn.Sequential(
            nn.Conv2d(chans_num_ouput, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )


        chans_num_input = int(1024 / 4)
        chans_num_token = int(1024 / 4)
        chans_num_ouput = int(1024 / 4)
        self.tokens_to_token1_1 = T2T_module(tokens_type='transformer', in_chans=chans_num_input,
                                             embed_dim=chans_num_token)
        self.tokens_to_token1_2 = T2T_module(tokens_type='transformer', in_chans=chans_num_token,
                                             embed_dim=chans_num_ouput)

        self.position_embedding1 = build_position_encoding(chans_num_input, scale=5 * 5)
        self.instnorm1_1 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_token),
            nn.ReLU()
        )
        self.instnorm1_2 = nn.Sequential(
            nn.InstanceNorm2d(chans_num_ouput),
            nn.ReLU()
        )

        self.conv_embed_1 = nn.Sequential(
            nn.Conv2d(chans_num_ouput*2, 256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.feat_merge_1 = nn.Sequential(
            nn.Conv2d(chans_num_ouput, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        chans_num_input=256
        chans_num_token=400
        self.tokens_interaction_1 = T2T_module_interaction(tokens_type='transformer', in_chans=chans_num_input,
                                              embed_dim=chans_num_token)
        # self.tokens_interaction_2 = T2T_module_interaction(tokens_type='transformer', in_chans=chans_num_input,
        #                                       embed_dim=chans_num_token)


       

        # self.conv_inter = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # # print('self.conv_inter.weight',self.conv_inter.weight)
        # if istrain==True:
        #     init=torch.Tensor([[[[0.0]],[[0.0]],[[1.0]],[[1.0]]]])
        #     self.conv_inter.weight=torch.nn.Parameter(init) 
        # # print('self.conv_inter.weight',self.conv_inter.weight)

        self.g_f=torch.nn.Parameter(torch.Tensor([[[[0.0]],[[0.0]],[[1.0]],[[1.0]]]]),requires_grad=True) 

        
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))

        
    def cam_normalize(self, x, d=1):
        eps = 1e-6
        norm = x.sum(dim=d, keepdim=True) + eps
        # print('norm.shape',norm.shape)
        return (x / norm)

        
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)


    def Max_normalize(self, x, d=1):
        max = torch.max(x,d,keepdim=True)[0] #max_value
        max=max.expand_as(x)
        return (x / max)


    def softmax_with_temperature(self, x, beta=100, d = 1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(beta*x)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    
    def forward(self, src_img, tgt_img, GT_src_mask = None, GT_tgt_mask = None, train=True):
        # Feature extraction
        src_feat1, src_feat2, src_feat3, src_feat4 = self.feature_extraction(src_img) # 256,80,80 // 512,40,40 // 1024,20,20 // 2048, 10, 10
        tgt_feat1, tgt_feat2, tgt_feat3, tgt_feat4 = self.feature_extraction(tgt_img)
        print('src_feat1.shape',src_feat1.shape)
        print('src_feat2.shape', src_feat2.shape)
        print('src_feat3.shape', src_feat3.shape)
        print('src_feat4.shape', src_feat4.shape)
        # Adaptation layers
        # src_feat3 = F.interpolate(src_feat3, scale_factor=2, mode='bilinear', align_corners=True)
        # tgt_feat3 = F.interpolate(tgt_feat3, scale_factor=2, mode='bilinear', align_corners=True)

        # src_feat3 = self.adap_layer_feat3(src_feat3)
        # tgt_feat3 = self.adap_layer_feat3(tgt_feat3)
        # src_feat4 = self.adap_layer_feat4(src_feat4)
        # src_feat4 = F.interpolate(src_feat4,scale_factor=2,mode='bilinear',align_corners=True)
        # tgt_feat4 = self.adap_layer_feat4(tgt_feat4)
        # tgt_feat4 = F.interpolate(tgt_feat4,scale_factor=2,mode='bilinear',align_corners=True)

        # print('src_feat1.shape',src_feat1.shape)
        # print('src_feat2.shape', src_feat2.shape)
        # print('src_feat3.shape', src_feat3.shape)
        # print('src_feat4.shape', src_feat4.shape)
        # os._exit()

        #Enhancement & Synergy

        src_feat4_adap = F.interpolate(src_feat4, scale_factor=0.5, mode='bilinear', align_corners=True)
        tgt_feat4_adap = F.interpolate(tgt_feat4, scale_factor=0.5, mode='bilinear', align_corners=True)

        pos = self.position_embedding4(src_feat4_adap)
        src_feat4_adap = self.instnorm4_1(self.tokens_to_token4_1(src_feat4_adap, pos))
        src_feat4_adap = self.instnorm4_2(self.tokens_to_token4_2(src_feat4_adap))
        tgt_feat4_adap = self.instnorm4_1(self.tokens_to_token4_1(tgt_feat4_adap, pos))
        tgt_feat4_adap = self.instnorm4_2(self.tokens_to_token4_2(tgt_feat4_adap))

        src_feat3_adap = F.interpolate(src_feat3, scale_factor=0.5, mode='bilinear', align_corners=True)
        tgt_feat3_adap = F.interpolate(tgt_feat3, scale_factor=0.5, mode='bilinear', align_corners=True)

        pos = self.position_embedding3(src_feat3_adap)
        src_feat3_adap = self.instnorm3_1(self.tokens_to_token3_1(src_feat3_adap, pos))
        src_feat3_adap = self.instnorm3_2(self.tokens_to_token3_2(src_feat3_adap))
        tgt_feat3_adap = self.instnorm3_1(self.tokens_to_token3_1(tgt_feat3_adap, pos))
        tgt_feat3_adap = self.instnorm3_2(self.tokens_to_token3_2(tgt_feat3_adap))

        src_feat2_adap = F.interpolate(src_feat2, scale_factor=0.5, mode='bilinear', align_corners=True)
        tgt_feat2_adap = F.interpolate(tgt_feat2, scale_factor=0.5, mode='bilinear', align_corners=True)

        pos = self.position_embedding2(src_feat2_adap)
        src_feat2_adap = self.instnorm2_1(self.tokens_to_token2_1(src_feat2_adap, pos))
        src_feat2_adap = self.instnorm2_2(self.tokens_to_token2_2(src_feat2_adap))
        tgt_feat2_adap = self.instnorm2_1(self.tokens_to_token2_1(tgt_feat2_adap, pos))
        tgt_feat2_adap = self.instnorm2_2(self.tokens_to_token2_2(tgt_feat2_adap))

        src_feat1_adap = F.interpolate(src_feat1, scale_factor=0.25, mode='bilinear', align_corners=True)
        tgt_feat1_adap = F.interpolate(tgt_feat1, scale_factor=0.25, mode='bilinear', align_corners=True)

        pos = self.position_embedding1(src_feat1_adap)
        src_feat1_adap = self.instnorm1_1(self.tokens_to_token1_1(src_feat1_adap, pos))
        src_feat1_adap = self.instnorm1_2(self.tokens_to_token1_2(src_feat1_adap))
        tgt_feat1_adap = self.instnorm1_1(self.tokens_to_token1_1(tgt_feat1_adap, pos))
        tgt_feat1_adap = self.instnorm1_2(self.tokens_to_token1_2(tgt_feat1_adap))



        # src_feat4_adap_avg_pooling=self.avgpooling(src_feat4_adap)
        # src_feat3_adap_avg_pooling=self.avgpooling(src_feat3_adap)
        # src_feat2_adap_avg_pooling=self.avgpooling(src_feat2_adap)
        # src_feat1_adap_avg_pooling=self.avgpooling(src_feat1_adap)

        # src_feat4_adap_cam=torch.sum(src_feat4_adap_avg_pooling*src_feat4_adap, dim=1).unsqueeze(1)
        # src_feat3_adap_cam=torch.sum(src_feat3_adap_avg_pooling*src_feat3_adap, dim=1).unsqueeze(1)
        # src_feat2_adap_cam=torch.sum(src_feat2_adap_avg_pooling*src_feat2_adap, dim=1).unsqueeze(1)
        # src_feat1_adap_cam=torch.sum(src_feat1_adap_avg_pooling*src_feat1_adap, dim=1).unsqueeze(1)
        # src_feat4_adap_cam=(src_feat4_adap_cam-torch.min(src_feat4_adap_cam))/(torch.max(src_feat4_adap_cam)
        #                     -torch.min(src_feat4_adap_cam))+0.5
        # src_feat3_adap_cam=(src_feat3_adap_cam-torch.min(src_feat3_adap_cam))/(torch.max(src_feat3_adap_cam)
        #                     -torch.min(src_feat3_adap_cam))+0.5
        # src_feat2_adap_cam=(src_feat2_adap_cam-torch.min(src_feat2_adap_cam))/(torch.max(src_feat2_adap_cam)
        #                     -torch.min(src_feat2_adap_cam))+0.5
        # src_feat1_adap_cam=(src_feat1_adap_cam-torch.min(src_feat1_adap_cam))/(torch.max(src_feat1_adap_cam)
        #                     -torch.min(src_feat1_adap_cam))+0.5

        
        tgt_feat4_adap_avg_pooling=self.avgpooling(tgt_feat4_adap)
        tgt_feat3_adap_avg_pooling=self.avgpooling(tgt_feat3_adap)
        tgt_feat2_adap_avg_pooling=self.avgpooling(tgt_feat2_adap)
        tgt_feat1_adap_avg_pooling=self.avgpooling(tgt_feat1_adap)

        tgt_feat4_adap_cam=torch.sum(tgt_feat4_adap_avg_pooling*tgt_feat4_adap, dim=1).unsqueeze(1)
        tgt_feat3_adap_cam=torch.sum(tgt_feat3_adap_avg_pooling*tgt_feat3_adap, dim=1).unsqueeze(1)
        tgt_feat2_adap_cam=torch.sum(tgt_feat2_adap_avg_pooling*tgt_feat2_adap, dim=1).unsqueeze(1)
        tgt_feat1_adap_cam=torch.sum(tgt_feat1_adap_avg_pooling*tgt_feat1_adap, dim=1).unsqueeze(1)
        tgt_feat4_adap_cam=(tgt_feat4_adap_cam-torch.min(tgt_feat4_adap_cam))/(torch.max(tgt_feat4_adap_cam)
                            -torch.min(tgt_feat4_adap_cam))+0.5
        tgt_feat3_adap_cam=(tgt_feat3_adap_cam-torch.min(tgt_feat3_adap_cam))/(torch.max(tgt_feat3_adap_cam)
                            -torch.min(tgt_feat3_adap_cam))+0.5
        tgt_feat2_adap_cam=(tgt_feat2_adap_cam-torch.min(tgt_feat2_adap_cam))/(torch.max(tgt_feat2_adap_cam)
                            -torch.min(tgt_feat2_adap_cam))+0.5
        tgt_feat1_adap_cam=(tgt_feat1_adap_cam-torch.min(tgt_feat1_adap_cam))/(torch.max(tgt_feat1_adap_cam)
                            -torch.min(tgt_feat1_adap_cam))+0.5
                        


        #Interaction
        # Correlation S2T
        corr_feat1 = self.matching_layer(src_feat1_adap, tgt_feat1_adap)  # channel : target / spatial grid : source
        corr_feat2 = self.matching_layer(src_feat2_adap, tgt_feat2_adap)  # channel : target / spatial grid : source
        corr_feat3 = self.matching_layer(src_feat3_adap, tgt_feat3_adap) # channel : target / spatial grid : source
        corr_feat4 = self.matching_layer(src_feat4_adap, tgt_feat4_adap)#b*hw*h*w
        print('corr_feat4.shape',corr_feat4.shape)


        #Interaction: constraint before interaction
        corr_S2T_1 = corr_feat1.clone()
        corr_S2T_1 = self.L2normalize(corr_S2T_1)
        corr_S2T_1 = self.apply_gaussian_kernel(corr_S2T_1)
        
        
        b, _, h, w = corr_feat1.size()
        corr_feat1 = corr_feat1.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)
        corr_T2S_1 = corr_feat1.clone()
        corr_T2S_1 = self.L2normalize(corr_T2S_1)
        corr_T2S_1 = self.apply_gaussian_kernel(corr_T2S_1)
        # #t2s->s2t
        # corr_feat1 = corr_feat1.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w) #b*hw*h*w

        corr_S2T_2 = corr_feat2.clone()
        corr_S2T_2 = self.L2normalize(corr_S2T_2)
        corr_S2T_2 = self.apply_gaussian_kernel(corr_S2T_2)
        
        b, _, h, w = corr_feat2.size()
        corr_feat2 = corr_feat2.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)
        corr_T2S_2 = corr_feat2.clone()
        corr_T2S_2 = self.L2normalize(corr_T2S_2)
        corr_T2S_2 = self.apply_gaussian_kernel(corr_T2S_2)
        # #t2s->s2t
        # corr_feat2 = corr_feat2.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w) #b*hw*h*w

        corr_S2T_3 = corr_feat3.clone()
        corr_S2T_3 = self.L2normalize(corr_S2T_3)
        corr_S2T_3 = self.apply_gaussian_kernel(corr_S2T_3)
        
        b, _, h, w = corr_feat3.size()
        corr_feat3 = corr_feat3.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)
        corr_T2S_3 = corr_feat3.clone()
        corr_T2S_3 = self.L2normalize(corr_T2S_3)
        corr_T2S_3 = self.apply_gaussian_kernel(corr_T2S_3)
        # #t2s->s2t
        # corr_feat3 = corr_feat3.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w) #b*hw*h*w

        corr_S2T_4 = corr_feat4.clone()
        corr_S2T_4 = self.L2normalize(corr_S2T_4)
        corr_S2T_4 = self.apply_gaussian_kernel(corr_S2T_4)
        
        b, _, h, w = corr_feat4.size()
        corr_feat4 = corr_feat4.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)
        corr_T2S_4 = corr_feat4.clone()
        corr_T2S_4 = self.L2normalize(corr_T2S_4)
        corr_T2S_4 = self.apply_gaussian_kernel(corr_T2S_4)
        # #t2s->s2t
        # corr_feat4 = corr_feat4.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w) #b*hw*h*w



        #Interaction: start
        tgt_cams=torch.cat((tgt_feat1_adap_cam,tgt_feat2_adap_cam,tgt_feat3_adap_cam,tgt_feat4_adap_cam),1)
        tgt_cams=tgt_cams*self.g_f
        tgt_cams=self.cam_normalize(tgt_cams)
        print('tgt_cams',tgt_cams)

        corr_feat_b,corr_feat_hw,corr_feat_h,corr_feat_w=corr_feat1.size()
        corr_feat1=corr_feat1.reshape(corr_feat_b*corr_feat_hw,corr_feat_h,corr_feat_w).unsqueeze(1)#b*1*hw*hw
        corr_feat2=corr_feat2.reshape(corr_feat_b*corr_feat_hw,corr_feat_h,corr_feat_w).unsqueeze(1)
        corr_feat3=corr_feat3.reshape(corr_feat_b*corr_feat_hw,corr_feat_h,corr_feat_w).unsqueeze(1)
        corr_feat4=corr_feat4.reshape(corr_feat_b*corr_feat_hw,corr_feat_h,corr_feat_w).unsqueeze(1)
        # print('corr_feat4.shape',corr_feat4.shape)
        corr_token=torch.cat((corr_feat1,corr_feat2,corr_feat3,corr_feat4),1)#b*4*hw*hw
        # print('corr_token.shape',corr_token.shape)



        corr_token=corr_token*tgt_cams
        corr_token=corr_token.sum(dim=1, keepdim=True)
        # print('corr_token.shape',corr_token.shape)
        print('self.g_f',self.g_f)

        # corr_feat1=corr_feat1*tgt_feat1_adap_cam
        # corr_feat2=corr_feat2*tgt_feat2_adap_cam
        # corr_feat3=corr_feat3*tgt_feat3_adap_cam
        # corr_feat4=corr_feat4*tgt_feat4_adap_cam

        # corr_feat_b,corr_feat_hw,corr_feat_h,corr_feat_w=corr_feat1.size()
        # corr_feat1=corr_feat1.reshape(corr_feat_b,corr_feat_hw,corr_feat_hw).unsqueeze(1)#b*1*hw*hw
        # corr_feat2=corr_feat2.reshape(corr_feat_b,corr_feat_hw,corr_feat_hw).unsqueeze(1)
        # corr_feat3=corr_feat3.reshape(corr_feat_b,corr_feat_hw,corr_feat_hw).unsqueeze(1)
        # corr_feat4=corr_feat4.reshape(corr_feat_b,corr_feat_hw,corr_feat_hw).unsqueeze(1)
        # print('corr_feat4.shape',corr_feat4.shape)
        # corr_token=torch.cat((corr_feat1,corr_feat2,corr_feat3,corr_feat4),1)#b*4*hw*hw
        # print('corr_token.shape',corr_token.shape)



        # corr_token=self.conv_inter(corr_token)
        # print('self.conv_inter.weight',self.conv_inter.weight)
        # print('self.conv_inter.bias',self.conv_inter.bias)

        # os._exit()

        corr_feat4=corr_token[:,0,:,:]
        print('corr_feat4.shape',corr_feat4.shape)
        corr_feat4=corr_feat4.reshape(corr_feat_b,corr_feat_hw,corr_feat_h,corr_feat_w)
        print('corr_feat4.shape',corr_feat4.shape)
      
        corr_T2S = corr_feat4.clone()
        corr_T2S = self.L2normalize(corr_T2S)    
        corr_T2S = self.apply_gaussian_kernel(corr_T2S)

        b, _, h, w = corr_feat4.size()
        corr_feat4 = corr_feat4.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)


        corr_S2T = corr_feat4.clone()
        corr_S2T = self.L2normalize(corr_S2T)
        corr_S2T = self.apply_gaussian_kernel(corr_S2T)

        # b, _, h, w = corr_feat4.size()
        # corr_feat4 = corr_feat4.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)
        
        # corr_T2S = corr_feat4.clone()
        # corr_T2S = self.L2normalize(corr_T2S)    
        # corr_T2S = self.apply_gaussian_kernel(corr_T2S)




                
        if not train:
            # Establish correspondences _1
            grid_S2T_1, flow_S2T_1, smoothness_S2T_1 = self.find_correspondence(corr_S2T_1)
            grid_T2S_1, flow_T2S_1, smoothness_T2S_1 = self.find_correspondence(corr_T2S_1)

            flow_T2S_1_clone=flow_T2S_1.clone()
            grid_S2T_1_clone=grid_S2T_1.clone()
            flow_S2T_1_clone = flow_S2T_1.clone()
            grid_T2S_1_clone = grid_T2S_1.clone()
            warped_flow_S2T_1 = -F.grid_sample(flow_T2S_1_clone, grid_S2T_1_clone)#warped_flow_S2T is opposite to flow_S2T
            warped_flow_T2S_1 = -F.grid_sample(flow_S2T_1_clone, grid_T2S_1_clone)

            # Establish correspondences _2
            grid_S2T_2, flow_S2T_2, smoothness_S2T_2 = self.find_correspondence(corr_S2T_2)
            grid_T2S_2, flow_T2S_2, smoothness_T2S_2 = self.find_correspondence(corr_T2S_2)

            flow_T2S_2_clone=flow_T2S_2.clone()
            grid_S2T_2_clone=grid_S2T_2.clone()
            flow_S2T_2_clone = flow_S2T_2.clone()
            grid_T2S_2_clone = grid_T2S_2.clone()
            warped_flow_S2T_2 = -F.grid_sample(flow_T2S_2_clone, grid_S2T_2_clone)#warped_flow_S2T is opposite to flow_S2T
            warped_flow_T2S_2 = -F.grid_sample(flow_S2T_2_clone, grid_T2S_2_clone)

            # Establish correspondences _3
            grid_S2T_3, flow_S2T_3, smoothness_S2T_3 = self.find_correspondence(corr_S2T_3)
            grid_T2S_3, flow_T2S_3, smoothness_T2S_3 = self.find_correspondence(corr_T2S_3)

            flow_T2S_3_clone=flow_T2S_3.clone()
            grid_S2T_3_clone=grid_S2T_3.clone()
            flow_S2T_3_clone = flow_S2T_3.clone()
            grid_T2S_3_clone = grid_T2S_3.clone()
            warped_flow_S2T_3 = -F.grid_sample(flow_T2S_3_clone, grid_S2T_3_clone)#warped_flow_S2T is opposite to flow_S2T
            warped_flow_T2S_3 = -F.grid_sample(flow_S2T_3_clone, grid_T2S_3_clone)

            # Establish correspondences _4
            grid_S2T_4, flow_S2T_4, smoothness_S2T_4 = self.find_correspondence(corr_S2T_4)
            grid_T2S_4, flow_T2S_4, smoothness_T2S_4 = self.find_correspondence(corr_T2S_4)

            flow_T2S_4_clone=flow_T2S_4.clone()
            grid_S2T_4_clone=grid_S2T_4.clone()
            flow_S2T_4_clone = flow_S2T_4.clone()
            grid_T2S_4_clone = grid_T2S_4.clone()
            warped_flow_S2T_4 = -F.grid_sample(flow_T2S_4_clone, grid_S2T_4_clone)#warped_flow_S2T is opposite to flow_S2T
            warped_flow_T2S_4 = -F.grid_sample(flow_S2T_4_clone, grid_T2S_4_clone)

            # Establish correspondences
            grid_S2T, flow_S2T, smoothness_S2T = self.find_correspondence(corr_S2T)
            grid_T2S, flow_T2S, smoothness_T2S = self.find_correspondence(corr_T2S)

            # warped_flow_S2T = -F.grid_sample(flow_T2S, grid_S2T, mode='bilinear')
            # warped_flow_T2S = -F.grid_sample(flow_S2T, grid_T2S, mode='bilinear')
            flow_T2S_clone=flow_T2S.clone()
            grid_S2T_clone=grid_S2T.clone()
            flow_S2T_clone = flow_S2T.clone()
            grid_T2S_clone = grid_T2S.clone()
            warped_flow_S2T = -F.grid_sample(flow_T2S_clone, grid_S2T_clone)#warped_flow_S2T is opposite to flow_S2T
            warped_flow_T2S = -F.grid_sample(flow_S2T_clone, grid_T2S_clone)
            
            return {'grid_S2T':grid_S2T, 'grid_T2S':grid_T2S, 'flow_S2T':flow_S2T, 'flow_T2S':flow_T2S,
                    'smoothness_S2T':smoothness_S2T,'smoothness_T2S':smoothness_T2S,
                    'warped_flow_S2T':warped_flow_S2T, 'warped_flow_T2S':warped_flow_T2S,
                    'grid_S2T_1':grid_S2T_1, 'grid_T2S_1':grid_T2S_1, 'flow_S2T_1':flow_S2T_1, 'flow_T2S_1':flow_T2S_1,
                    'smoothness_S2T_1':smoothness_S2T_1,'smoothness_T2S_1':smoothness_T2S_1,
                    'warped_flow_S2T_1':warped_flow_S2T_1, 'warped_flow_T2S_1':warped_flow_T2S_1,
                    'grid_S2T_2':grid_S2T_2, 'grid_T2S_2':grid_T2S_2, 'flow_S2T_2':flow_S2T_2, 'flow_T2S_2':flow_T2S_2,
                    'smoothness_S2T_2':smoothness_S2T_2,'smoothness_T2S_2':smoothness_T2S_2,
                    'warped_flow_S2T_2':warped_flow_S2T_2, 'warped_flow_T2S_2':warped_flow_T2S_2,
                    'grid_S2T_3':grid_S2T_3, 'grid_T2S_3':grid_T2S_3, 'flow_S2T_3':flow_S2T_3, 'flow_T2S_3':flow_T2S_3,
                    'smoothness_S2T_3':smoothness_S2T_3,'smoothness_T2S_3':smoothness_T2S_3,
                    'warped_flow_S2T_3':warped_flow_S2T_3, 'warped_flow_T2S_3':warped_flow_T2S_3,
                    'grid_S2T_4':grid_S2T_4, 'grid_T2S_4':grid_T2S_4, 'flow_S2T_4':flow_S2T_4, 'flow_T2S_4':flow_T2S_4,
                    'smoothness_S2T_4':smoothness_S2T_4,'smoothness_T2S_4':smoothness_T2S_4,
                    'warped_flow_S2T_4':warped_flow_S2T_4, 'warped_flow_T2S_4':warped_flow_T2S_4,}
        else:
            # Establish correspondences
            grid_S2T, flow_S2T, smoothness_S2T = self.find_correspondence(corr_S2T, GT_src_mask)
            grid_T2S, flow_T2S, smoothness_T2S = self.find_correspondence(corr_T2S, GT_tgt_mask)
            
            # Estimate warped masks
            warped_src_mask = F.grid_sample(GT_tgt_mask, grid_S2T, mode = 'bilinear')
            warped_tgt_mask = F.grid_sample(GT_src_mask, grid_T2S, mode = 'bilinear')
            
            # Estimate warped flows
            warped_flow_S2T = -F.grid_sample(flow_T2S, grid_S2T, mode = 'bilinear') * GT_src_mask
            warped_flow_T2S = -F.grid_sample(flow_S2T, grid_T2S, mode = 'bilinear') * GT_tgt_mask
            flow_S2T = flow_S2T * GT_src_mask
            flow_T2S = flow_T2S * GT_tgt_mask

            return {'est_src_mask':warped_src_mask, 'smoothness_S2T':smoothness_S2T, 'grid_S2T':grid_S2T,
                    'est_tgt_mask':warped_tgt_mask, 'smoothness_T2S':smoothness_T2S, 'grid_T2S':grid_T2S, 
                    'flow_S2T':flow_S2T, 'flow_T2S':flow_T2S,
                    'warped_flow_S2T':warped_flow_S2T, 'warped_flow_T2S':warped_flow_T2S}
