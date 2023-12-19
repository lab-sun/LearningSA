# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util
import os
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.model = models.resnet101(pretrained=True)
        resnet_feature_layers = ['conv1',
                                 'bn1',
                                 'relu',
                                 'maxpool',
                                 'layer1',
                                 'layer2',
                                 'layer3',
                                 'layer4',]
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
                              self.model.layer4]
        self.layer1 = nn.Sequential(*resnet_module_list[:layer1_idx + 1])
        self.layer2 = nn.Sequential(*resnet_module_list[layer1_idx + 1:layer2_idx + 1])
        self.layer3 = nn.Sequential(*resnet_module_list[layer2_idx + 1:layer3_idx + 1])
        self.layer4 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False

    def forward(self, image_batch):
        layer1_feat = self.layer1(image_batch)
        layer2_feat = self.layer2(layer1_feat)
        layer3_feat = self.layer3(layer2_feat)
        layer4_feat = self.layer4(layer3_feat)
        return layer1_feat, layer2_feat, layer3_feat, layer4_feat

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        # super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])  #r11
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])  #r21
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])  #r31
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])  #r41
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])  #r51
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self,  vgg_normal_correct=False):
        super(VGGLoss, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct
        # if vgg_normal_correct:
        #     self.vgg = VGG19_feature_color_torchversion(vgg_normal_correct=True).cuda()
        # else:
        #     self.vgg = VGG19().cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = VGG19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        # if self.vgg_normal_correct:
        #     x_vgg, y_vgg = self.vgg(x, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True), self.vgg(y, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True)
        # else:
        #     x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.alpha = 1

        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

        #vgg initialization
        self.vggnet_fix = networks.correspondence.VGG19_feature_color_torchversion(
            vgg_normal_correct=opt.vgg_normal_correct)
        # self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
        self.vggnet_fix.load_state_dict(torch.load(opt.vgg_path))
        self.vggnet_fix.eval()
        for param in self.vggnet_fix.parameters():
            param.requires_grad = False

        self.vggnet_fix.to(self.opt.gpu_ids[0])

        self.VGGLoss = VGGLoss()
        self.VGGLoss.eval()

        self.feature_extraction = FeatureExtraction()
        self.feature_extraction.to(self.opt.gpu_ids[0])
        self.feature_extraction.eval()

        # set loss functions
        if opt.isTrain:
            # self.vggnet_fix = networks.correspondence.VGG19_feature_color_torchversion(vgg_normal_correct=opt.vgg_normal_correct)
            # # self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
            # self.vggnet_fix.load_state_dict(torch.load(opt.vgg_path))
            # self.vggnet_fix.eval()
            # for param in self.vggnet_fix.parameters():
            #     param.requires_grad = False
            #
            # self.vggnet_fix.to(self.opt.gpu_ids[0])
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.MSE_loss = torch.nn.MSELoss()
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, GforD=None, alpha=1,logs_writer=None,batch_i=0):
        # print('data',data)
        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics_r,ref_semantics_g,ref_semantics_b,ref_semantics,photo,photo_semantics,flow_aff2photo,image_no_norm,ref_no_norm = self.preprocess_input(data, )
        # print('input_label.size',input_label.size())
        # print('input_semantics.size', input_semantics.size())
        # print('real_image.size', real_image.size())
        # print('self_ref.size', self_ref.size())
        # print('ref_image.size', ref_image.size())
        # print('ref_label.size', ref_label.size())
        # print('ref_semantics.size', ref_semantics.size())

        self.alpha = alpha
        generated_out = {}
        if mode == 'generator':
            # print('ref_image',ref_image)
            g_loss, generated_out = self.compute_generator_loss(input_label, 
                input_semantics, real_image, ref_label, ref_semantics_r,ref_semantics_g,ref_semantics_b,ref_semantics, ref_image, self_ref,photo,photo_semantics,flow_aff2photo)
            
            out = {}
            # out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['warp_mask'] = None if 'warp_mask' not in generated_out else generated_out['warp_mask']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in generated_out else generated_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_feature_img' not in generated_out else generated_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            out['warp_i2r'] = None if 'warp_i2r' not in generated_out else generated_out['warp_i2r']
            out['warp_i2r2i'] = None if 'warp_i2r2i' not in generated_out else generated_out['warp_i2r2i']
            out['warped_src'] = None if 'warped_src' not in generated_out else generated_out['warped_src']
            out['warped_tgt'] = None if 'warped_tgt' not in generated_out else generated_out['warped_tgt']
            out['patch_warp_out'] = None if 'patch_warp_out' not in generated_out else generated_out['patch_warp_out']
            out['patch_warp_out_upsample'] = None if 'patch_warp_out_upsample' not in generated_out else generated_out['patch_warp_out_upsample']
            out['warped_photo'] = None if 'warped_photo' not in generated_out else generated_out['warped_photo']
            out['photo'] = None if 'photo' not in generated_out else generated_out['photo']
            out['ref_img'] = None if 'ref_img' not in generated_out else generated_out['ref_img']
            out['real_img'] = None if 'real_img' not in generated_out else generated_out['real_img']
            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, GforD, label=input_label)
            return d_loss
        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics, real_img=real_image, ref_semantics_r=ref_semantics_r, ref_semantics_g=ref_semantics_g, ref_semantics_b=ref_semantics_b,
                        ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref,photo = photo, photo_semantics = photo_semantics,input_label=input_label,ref_no_norm=ref_no_norm,
                                     logs_writer=logs_writer,batch_i=batch_i)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out
        else:
            raise ValueError("|mode| is invalid")

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    # def lossfn_two_var(self, target1, target2, num_px = None):
    #     if num_px is None:
    #         return torch.sum(torch.pow((target1 - target2),2))
    #     else:
    #         return torch.sum(torch.pow((target1 - target2),2) / num_px)
    def lossfn_two_var(self, target1, target2, num_px = None):
        if num_px is None:
            data1 = torch.pow((target1 - target2), 2)  # b*n*h*w
            data2 = torch.sum(data1, dim=1)  # b*h*w
            data2 = data2.sqrt()
            return torch.sum(data2)
        else:
            data1=torch.pow((target1 - target2),2)#b*n*h*w
            data2 = torch.sum(data1, dim=1)#b*h*w
            data2=data2.sqrt()
            return torch.sum(data2 / num_px)

    def create_optimizers(self, opt):
        G_params, D_params = list(), list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr*0.5}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr*0.5}]
        if opt.isTrain:
            D_params += list(self.net['netD'].parameters())
            if opt.weight_domainC > 0 and opt.domain_rela:
                D_params += list(self.net['netDomainClassifier'].parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netD'], 'D', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)
        if self.opt.weight_domainC > 0 and self.opt.domain_rela: 
            util.save_network(self.net['netDomainClassifier'], 'DomainClassifier', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netD'] = networks.define_D(opt) if opt.isTrain else None
        net['netCorr'] = networks.define_Corr(opt)
        net['netDomainClassifier'] = networks.define_DomainClassifier(opt) if opt.weight_domainC > 0 and opt.domain_rela else None

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)
            if opt.weight_domainC > 0 and opt.domain_rela:
                net['netDomainClassifier'] = util.load_network(net['netDomainClassifier'], 'DomainClassifier', opt.which_epoch, opt)
            if (not opt.isTrain) and opt.use_ema:
                net['netG'] = util.load_network(net['netG'], 'G_ema', opt.which_epoch, opt)
                net['netCorr'] = util.load_network(net['netCorr'], 'netCorr_ema', opt.which_epoch, opt)
        return net
        #return netG_stage1, netD_stage1, netG, netD, netE, netCorr

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # print('data[label]', data['label'])
        # print('data[image]', data['image'])
        # print('data[self_ref]', data['self_ref'])
        # print('data[ref]', data['ref'])
        # print('data[label_ref]', data['label_ref'])

        #adapt some component in data
        data['photo'] = data['ref'].clone().cuda()
        # _, _, _, photo_semantics, _ = self.vggnet_fix(data['ref'].cuda(), ['r12', 'r22', 'r32', 'r42', 'r52'],preprocess=False)
        # # photo_semantics = F.interpolate(photo_semantics, scale_factor=8, mode='nearest')
        # # print(data['photo'] - data['ref'].cuda())

        _, _, photo_feat3, photo_feat4 = self.feature_extraction(data['ref'].cuda())


        data['image']=data['label_norm'].clone()
        data['ref'] = data['label_ref_norm_aff'].clone()
        data['label_ref'] = data['label_ref_aff'].clone()

        # print(data['photo']-data['ref'].cuda())


        # label_feat=self.vggnet_fix(data['label_norm'].cuda(), ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        # label_ref_aff_feat = self.vggnet_fix(data['label_ref_norm_aff'].cuda(), ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        #
        # _,_,_,label_feat_4,_ = self.vggnet_fix(data['label_norm'].cuda(), ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        #
        # print('label_feat', type(label_feat))
        # for i in range(5):
        #     print('label_feat',label_feat[i].shape)
        # print('label_feat_4', label_feat_4.shape)
        #
        # label_feat_4 = F.interpolate(label_feat_4, scale_factor=8, mode='nearest')
        # print('label_feat_4', label_feat_4.shape)
        #
        # print('data[label_norm].size()',data['label_norm'].size())
        # print('data[label_ref_norm_aff].size()', data['label_ref_norm_aff'].size())
        # os._exit()


        if self.opt.dataset_mode == 'celebahq':
            print(1)
            glasses = data['label'][:,1::2,:,:].long()
            data['label'] = data['label'][:,::2,:,:]
            glasses_ref = data['label_ref'][:,1::2,:,:].long()
            data['label_ref'] = data['label_ref'][:,::2,:,:]
            if self.use_gpu():
                print(2)
                glasses = glasses.cuda()
                glasses_ref = glasses_ref.cuda()
        elif self.opt.dataset_mode == 'celebahqedge':
            print(3)
            input_semantics = data['label'].clone().cuda().float()
            data['label'] = data['label'][:,:1,:,:]
            ref_semantics = data['label_ref'].clone().cuda().float()
            data['label_ref'] = data['label_ref'][:,:1,:,:]
        elif self.opt.dataset_mode == 'deepfashion':
            print(4)
            input_semantics = data['label'].clone().cuda().float()
            data['label'] = data['label'][:,:3,:,:]
            ref_semantics = data['label_ref'].clone().cuda().float()
            data['label_ref'] = data['label_ref'][:,:3,:,:]

        # move to GPU and change data types
        if self.opt.dataset_mode != 'deepfashion':
            print(5)
            data['label'] = data['label'].long()
            # print(data['label'])
        if self.use_gpu():
            print(6)
            data['label'] = data['label'].cuda()
            # print(data['label'])
            data['image'] = data['image'].cuda()
            # print(data['image_no_norm'])
            data['ref'] = data['ref'].cuda()
            data['label_ref'] = data['label_ref'].cuda()
            if self.opt.dataset_mode != 'deepfashion':
                print(7)
                data['label_ref'] = data['label_ref'].long()
            data['self_ref'] = data['self_ref'].cuda()

        # create one-hot label map
        if self.opt.dataset_mode != 'celebahqedge' and self.opt.dataset_mode != 'deepfashion':
            print(8)
            label_map = data['label']
            bs, _, h, w = label_map.size()
            # nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            #     else self.opt.label_nc
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            # input_label = self.FloatTensor(bs, nc, h, w).zero_()
            # input_semantics = input_label.scatter_(1, label_map, 1.0)
            #
            # label_map = data['label_ref']
            # label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            # ref_semantics = label_ref.scatter_(1, label_map, 1.0)


            # _, _, _, input_semantics, _ = self.vggnet_fix(data['label_norm'].cuda(), ['r12', 'r22', 'r32', 'r42', 'r52'],preprocess=False)
            # # input_semantics = F.interpolate(input_semantics, scale_factor=8, mode='nearest')
            # _, _, _, ref_semantics, _ = self.vggnet_fix(data['label_ref_norm_aff'].cuda(), ['r12', 'r22', 'r32', 'r42', 'r52'],preprocess=False)
            # # ref_semantics = F.interpolate(ref_semantics, scale_factor=8, mode='nearest')

            # os._exit()

            _, _, src_feat3, src_feat4 = self.feature_extraction(data['label_norm'].cuda())
            _, _, tgt_feat3, tgt_feat4 = self.feature_extraction(data['label_ref_norm_aff'].cuda())



            #ref_semantics component with r,g,b
            label_map = data['label_ref'][:,0,:,:].unsqueeze(1)
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics_r = label_ref.scatter_(1, label_map, 1.0)
            label_map = data['label_ref'][:, 1, :, :].unsqueeze(1)
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics_g = label_ref.scatter_(1, label_map, 1.0)
            label_map = data['label_ref'][:, 2, :, :].unsqueeze(1)
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics_b = label_ref.scatter_(1, label_map, 1.0)



            # #using images as labels
            # label_map = data['image_no_norm'].long().cuda()
            # bs, _, h, w = label_map.size()
            # # nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            # #     else self.opt.label_nc
            # nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            #     else self.opt.label_nc
            # input_label = self.FloatTensor(bs, nc, h, w).zero_()
            # input_semantics = input_label.scatter_(1, label_map, 1.0)
            #
            # label_map = data['ref_no_norm'].long().cuda()
            # label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            # ref_semantics = label_ref.scatter_(1, label_map, 1.0)


        # # print('label_map', label_map)
        # print('label_map',label_map)
        # print('label_map.size', label_map.size())
        # # print('input_semantics',input_semantics)
        # print('input_semantics,size', input_semantics.size())
        # # print(data['image'])
        # # print(input_semantics[0, :, 0, 0])
        # print(torch.sum(input_semantics[0,:,0,0]))
        # os._exit()

        if self.opt.dataset_mode == 'celebahq':
            print(9)
            assert input_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            input_semantics[:,-3:-2,:,:] = glasses
            assert ref_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            ref_semantics[:,-3:-2,:,:] = glasses_ref
        # print('data[label]',data['label'])
        # print('input_semantics', input_semantics)
        # print('data[image]', data['image'])
        # print('data[self_ref]', data['self_ref'])
        # print('data[ref]', data['ref'])
        # print('data[label_ref]', data['label_ref'])
        # print('ref_semantics', ref_semantics)
        data['flow_aff2ref']=data['flow_aff2ref'].cuda()

        if not self.opt.isTrain:
            data['ref']=data['photo']
            photo_semantics = photo_feat3
            input_semantics = src_feat3
            ref_semantics = photo_feat3


        if self.opt.isTrain:
            photo_semantics = photo_feat3
            input_semantics = src_feat3
            ref_semantics = tgt_feat3

        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics_r,ref_semantics_g,ref_semantics_b,ref_semantics,data['photo'],photo_semantics,data['flow_aff2ref'],data['image_no_norm'],data['ref_no_norm']

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-4], 4), F.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics_r=None,ref_semantics_g=None,ref_semantics_b=None,ref_semantics=None, ref_image=None, self_ref=None,photo=None,photo_semantics=None,flow_aff2photo=None):
        G_losses = {}
        generate_out = self.generate_fake(
            input_semantics, real_image, ref_semantics_r,ref_semantics_g,ref_semantics_b,ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref,photo=photo,photo_semantics=photo_semantics)

        batchsize, c, image_h, image_w = real_image.size()

        # if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
        #     G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
        #     G_losses['no_vgg_feat']=0.0
        #
        # if self.opt.warp_cycle_w > 0:
        #     if not self.opt.warp_patch:
        #         ref = F.avg_pool2d(ref_image, self.opt.warp_stride)
        #     else:
        #         ref = ref_image
        #
        #     G_losses['G_warp_cycle'] = F.l1_loss(generate_out['warp_cycle'], ref) * self.opt.warp_cycle_w
        #     if self.opt.two_cycle:
        #         real = F.avg_pool2d(real_image, self.opt.warp_stride)
        #         G_losses['G_warp_cycle'] += F.l1_loss(generate_out['warp_i2r2i'], real) * self.opt.warp_cycle_w
        #
        # if self.opt.warp_self_w > 0:
        #     sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #     G_losses['G_warp_self'] = torch.mean(F.l1_loss(generate_out['warp_out'], real_image, reduce=False) * sample_weights) * self.opt.warp_self_w
        #
        #
        # pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
        #     input_semantics, generate_out['fake_image'], real_image)
        #
        # G_losses['GAN'] = self.criterionGAN(pred_fake, True,
        #                                     for_discriminator=False) * self.opt.weight_gan
        # G_losses['GAN']=0.0
        #
        # if not self.opt.no_ganFeat_loss:
        #     num_D = len(pred_fake)
        #     GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        #     for i in range(num_D):  # for each discriminator
        #         # last output is the final prediction, so we exclude it
        #         num_intermediate_outputs = len(pred_fake[i]) - 1
        #         for j in range(num_intermediate_outputs):  # for each layer output
        #             unweighted_loss = self.criterionFeat(
        #                 pred_fake[i][j], pred_real[i][j].detach())
        #             GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        #     G_losses['GAN_Feat'] = GAN_Feat_loss
        #     G_losses['GAN_Feat']=0.0
        #
        # fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        # sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # loss = 0
        # for i in range(len(generate_out['real_features'])):
        #     loss += weights[i] * util.weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(), sample_weights)
        # G_losses['fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio
        # G_losses['fm']=0.0
        #
        # feat_loss = util.mse_loss(fake_features[self.perceptual_layer], generate_out['real_features'][self.perceptual_layer].detach())
        # G_losses['perc'] = feat_loss * self.opt.weight_perceptual
        # G_losses['perc'] = 0.0
        #
        # G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out['ref_features']) * self.opt.lambda_vgg * self.opt.ctx_w
        # G_losses['contextual'] = 0.0

        # if self.opt.warp_mask_losstype != 'none':
        #     # ref_label = F.interpolate(ref_label.float(), scale_factor=0.25, mode='nearest').long().squeeze(1)
        #     # gt_label = F.interpolate(input_label.float(), scale_factor=0.25, mode='nearest').long().squeeze(1)
        #     # weights = []
        #     # for i in range(ref_label.shape[0]):
        #     #     ref_label_uniq = torch.unique(ref_label[i])
        #     #     gt_label_uniq = torch.unique(gt_label[i])
        #     #     zero_label = [it for it in gt_label_uniq if it not in ref_label_uniq]
        #     #     weight = torch.ones_like(gt_label[i]).float()
        #     #     for j in zero_label:
        #     #         weight[gt_label[i] == j] = 0
        #     #     weight[gt_label[i] == 0] = 0 #no loss from unknown class
        #     #     weights.append(weight.unsqueeze(0))
        #     # weights = torch.cat(weights, dim=0)
        #
        #
        #     # input_semantics_reduce=F.interpolate(input_semantics, scale_factor=0.25, mode='nearest')
        #     # warp_feat_loss = util.mse_loss(generate_out['warp_mask'],input_semantics_reduce)
        #     # G_losses['mask'] = warp_feat_loss * self.opt.weight_perceptual
        #
        #     # print('input_semantics.shape',input_semantics.shape)
        #     # print('generate_out[warp_mask].shape', generate_out['warp_mask'].shape)
        #     input_semantics_reduce = F.interpolate(input_semantics, scale_factor=4, mode='nearest')
        #     warp_feat_loss = util.mse_loss(self.L2normalize(generate_out['warp_mask']), self.L2normalize(input_semantics_reduce))
        #     G_losses['mask'] = warp_feat_loss * 512*0.0
        #
        #     print('mask',G_losses['mask'])
        #     # print(generate_out['warp_mask']-input_semantics_reduce)
        #     # print('generate_out[warp_mask].shape',generate_out['warp_mask'].shape)
        #     # print('input_semantics_reduce.shape', input_semantics_reduce.shape)
        #     # print('generate_out[warp_mask]', generate_out['warp_mask'])
        #     # print('input_semantics_reduce', input_semantics_reduce)
        #     # os._exit()

        cross_domain=True
        if cross_domain:
            # # print('flow_aff2photo.shape',flow_aff2photo.shape)
            # flow_aff2photo = F.interpolate(flow_aff2photo, scale_factor=0.25, mode='nearest')
            # # print('flow_aff2photo.shape', flow_aff2photo.shape)
            # # print('flow_aff2photo.shape', generate_out['flow_T2P'].shape)
            # flow_loss = util.mse_loss(generate_out['flow_T2P'], flow_aff2photo)
            # G_losses['flow'] = flow_loss * 2
            # # os._exit()
            # print('generate_out[flow_T2P].shape',generate_out['flow_T2P'].shape)
            flow_T2P = F.interpolate(generate_out['flow_T2P'], scale_factor=8, mode='nearest')
            flow_loss = self.lossfn_two_var(flow_T2P * image_h / 2, flow_aff2photo * image_h / 2,
                                            image_h * image_w)  # flow loss
            G_losses['flow'] = flow_loss

            print('flow',G_losses['flow'])

        perc=True
        if perc:
            # # print('generate_out[warped_tgt]',generate_out['warped_tgt'])
            # # print('real_image', real_image)
            # maximum_warping_features = self.vggnet_fix(generate_out['warped_tgt'], ['r12', 'r22', 'r32', 'r42', 'r52'],preprocess=False)
            # # maximum_warping_features = self.vggnet_fix(generate_out['warp_out'], ['r12', 'r22', 'r32', 'r42', 'r52'],
            # #                                            preprocess=True)
            # real_image_vgg_features = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'],preprocess=False)
            # weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            # loss = 0
            # for i in range(len(real_image_vgg_features)):
            #     loss += weights[i] * util.l1_loss(self.L2normalize(maximum_warping_features[i]), self.L2normalize(real_image_vgg_features[i]))
            # G_losses['perc_maximum'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio*20
            # print('perc_maximum', G_losses['perc_maximum'])
            #
            # average_warping_features = self.vggnet_fix(generate_out['warp_out'], ['r12', 'r22', 'r32', 'r42', 'r52'],
            #                                            preprocess=False)
            # loss2 = 0
            # for i in range(len(real_image_vgg_features)):
            #     loss2 += weights[i] * util.l1_loss(self.L2normalize(average_warping_features[i]), self.L2normalize(real_image_vgg_features[i]))
            # G_losses['perc_average'] = loss2 * self.opt.lambda_vgg * self.opt.fm_ratio*20*0.0
            # print('perc_average', G_losses['perc_average'])

            perc_maximum = self.VGGLoss(real_image, generate_out['warped_tgt'])
            G_losses['perc_maximum'] = perc_maximum * 10
            print('perc_maximum', G_losses['perc_maximum'])

            # perc_average = self.VGGLoss(real_image, generate_out['warp_out'])
            # G_losses['perc_average'] = perc_average * 0.0
            # print('perc_average', G_losses['perc_average'])

            # os._exit()

        recons=True
        if recons:
            G_losses['recons_maximum'] = util.l1_loss(real_image, generate_out['warped_tgt']) * 10
            # G_losses['recons_average'] = util.l1_loss(real_image, generate_out['warp_out']) * 0.0
            print('recons_maximum', G_losses['recons_maximum'])
            # print('recons_average', G_losses['recons_average'])

        consist = True
        if consist:
            # calculate flow consistency loss
            _, _, flow_h, flow_w = generate_out['flow_S2T'].size()
            # G_losses['consist_S2T'] = self.lossfn_two_var(generate_out['flow_S2T']* image_h / 2, generate_out['warped_flow_T2S']* image_h / 2,
            #                          flow_h*flow_w)  # flow consistency
            # G_losses['consist_T2P'] = self.lossfn_two_var(generate_out['flow_T2P']* image_h / 2, generate_out['warped_flow_P2T']* image_h / 2,
            #                                                flow_h*flow_w)*0.5   # flow consistency
            # print('consist_S2T', G_losses['consist_S2T'])
            # print('consist_T2P', G_losses['consist_T2P'])

            G_losses['consist_S2T'] = util.l1_loss(generate_out['flow_S2T'] * image_h / 2,
                                                   generate_out['warped_flow_T2S'] * image_h / 2) * 0.5
            consist_T2P = self.lossfn_two_var(generate_out['flow_T2P'] * image_h / 2,
                                              generate_out['warped_flow_P2T'] * image_h / 2,
                                              flow_h * flow_w)  # flow consistency
            print('consist_S2T', G_losses['consist_S2T'])
            print('consist_T2P', consist_T2P)

        smoothness = False
        if smoothness:
            # smoothness metric   average flow gradient(horizontal and vertical direction) for each flow map
            smoothness_T2S = generate_out['smoothness_T2S']  # b * c * 64*64
            # smoothness_T2P = generate_out['smoothness_T2P']
            # smoothness_P2T = generate_out['smoothness_P2T']
            smoothness_S2T = generate_out['smoothness_S2T']

            smoothness_b, smoothness_c, smoothness_h, smoothness_w = smoothness_T2S.size()
            smoothness_T2S_loss = (torch.sum(smoothness_T2S) * image_h / 2.0 / (
                        smoothness_h * smoothness_w) / 2)  # accumulate average flow gradient of each flow map
            # smoothness_T2P_loss = (torch.sum(smoothness_T2P) * image_h / 2.0 / (
            #         smoothness_h * smoothness_w)/2)*0.0  # accumulate average flow gradient of each flow map
            # smoothness_P2T_loss = (torch.sum(smoothness_P2T) * image_h / 2.0 / (
            #         smoothness_h * smoothness_w)/2)*0.0  # accumulate average flow gradient of each flow map
            smoothness_S2T_loss = (torch.sum(smoothness_S2T) * image_h / 2.0 / (
                    smoothness_h * smoothness_w) / 2)  # accumulate average flow gradient of each flow map in average (x and y) directions
            # print('smoothness_T2S_loss',smoothness_T2S_loss)
            # print('smoothness_T2P_loss', smoothness_T2P_loss)
            # print('smoothness_P2T_loss', smoothness_P2T_loss)
            # print('smoothness_S2T_loss', smoothness_S2T_loss)
            # os._exit()
            # G_losses['smoothness']=(smoothness_T2S_loss+smoothness_T2P_loss+smoothness_P2T_loss+smoothness_S2T_loss)*0.25*0.0
            G_losses['smoothness'] = (smoothness_T2S_loss + smoothness_S2T_loss) * 0.5 * 0.25
            print('smoothness_T2S_loss', smoothness_T2S_loss)
            # print('smoothness_T2P_loss', smoothness_T2P_loss)
            # print('smoothness_P2T_loss', smoothness_P2T_loss)
            print('smoothness_S2T_loss', smoothness_S2T_loss)
            print('G_losses[smoothness]', G_losses['smoothness'])



        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, GforD, label=None):
        D_losses = {}
        with torch.no_grad():
            #fake_image, _, _, _, _ = self.generate_fake(input_semantics, real_image, VGG_feat=False)
            fake_image = GforD['fake_image'].detach()
            fake_image.requires_grad_()
            
        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                            for_discriminator=True) * self.opt.weight_gan

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.net['netE'](real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, ref_semantics_r,ref_semantics_g,ref_semantics_b,ref_semantics=None, ref_image=None, self_ref=None,photo=None,photo_semantics=None):
        generate_out = {}
        #print(ref_image.max())
        # print('ref_image',ref_image)
        # ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1 = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        
        coor_out = self.net['netCorr'](ref_image, real_image, input_semantics, ref_semantics_r,ref_semantics_g,ref_semantics_b,ref_semantics, alpha=self.alpha,photo=photo,photo_semantics=photo_semantics)
        
        # generate_out['ref_features'] = [ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1]
        # generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)

        # os._exit()
        
        # if self.opt.CBN_intype == 'mask':
        #     CBN_in = input_semantics
        #     # print(1)
        #     # print('CBN_in.size',CBN_in.size())
        # # elif self.opt.CBN_intype == 'warp':
        # #     CBN_in = coor_out['warp_out']
        # #     # print(2)
        # #     # print('CBN_in.size', CBN_in.size())
        # elif self.opt.CBN_intype == 'warp':  #using our warped operation
        #     CBN_in = coor_out['warped_tgt']
        # elif self.opt.CBN_intype == 'warp_mask':
        #     CBN_in = torch.cat((coor_out['warp_out'], input_semantics), dim=1)
        #     # print(3)
        #     # print('CBN_in.size', CBN_in.size())
        # # elif self.opt.CBN_intype == 'warp_mask':
        # #     CBN_in = torch.cat((coor_out['warped_tgt'], input_semantics), dim=1)
        # #     # print(3)
        # #     # print('CBN_in.size', CBN_in.size())
        #
        # # os._exit()
        #
        # # generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=CBN_in)


        generate_out = {**generate_out, **coor_out}
        return generate_out

    def inference(self, input_semantics,real_img=None, ref_semantics_r=None, ref_semantics_g=None, ref_semantics_b=None,ref_semantics=None, ref_image=None, self_ref=None,photo = None, photo_semantics = None,input_label=None,ref_no_norm=None,logs_writer=None,batch_i=0):
        generate_out = {}

        img_grid = vutils.make_grid(real_img, nrow=6, padding=0, normalize=True, scale_each=True)
        logs_writer.add_image('real_img', img_grid, global_step=batch_i)

        img_grid = vutils.make_grid(ref_image, nrow=6, padding=0, normalize=True, scale_each=True)
        logs_writer.add_image('ref_image', img_grid, global_step=batch_i)

        coor_out = self.net['netCorr'](ref_image, real_img, input_semantics, ref_semantics_r, ref_semantics_g, ref_semantics_b,ref_semantics, alpha=self.alpha,photo = photo, photo_semantics = photo_semantics,input_label=input_label,ref_no_norm=ref_no_norm,logs_writer=logs_writer,batch_i=batch_i)
        # if self.opt.CBN_intype == 'mask':
        #     CBN_in = input_semantics
        # elif self.opt.CBN_intype == 'warp':
        #     CBN_in = coor_out['warp_out']
        # elif self.opt.CBN_intype == 'warp_mask':
        #     CBN_in = torch.cat((coor_out['warp_out'], input_semantics), dim=1)
        #
        # generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=CBN_in)
        # generate_out = {**generate_out, **coor_out}
        generate_out = {**coor_out}
        return generate_out

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        seg = None
        discriminator_out, seg, cam_logit = self.net['netD'](fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)
        fake_cam_logit, real_cam_logit = None, None
        if self.opt.D_cam > 0:
            fake_cam_logit = torch.cat([it[:it.shape[0]//2] for it in cam_logit], dim=1)
            real_cam_logit = torch.cat([it[it.shape[0]//2:] for it in cam_logit], dim=1)
        #fake_cam_logit, real_cam_logit = self.divide_pred(cam_logit)

        return pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_D_seg_loss(self, out, gt):
        fake_seg, real_seg = self.divide_pred([out])
        fake_seg_loss = F.cross_entropy(fake_seg[0][0], gt)
        real_seg_loss = F.cross_entropy(real_seg[0][0], gt)

        down_gt = F.interpolate(gt.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze().long()
        fake_seg_loss_down = F.cross_entropy(fake_seg[0][1], down_gt)
        real_seg_loss_down = F.cross_entropy(real_seg[0][1], down_gt)

        seg_loss = fake_seg_loss + real_seg_loss + fake_seg_loss_down + real_seg_loss_down
        return seg_loss