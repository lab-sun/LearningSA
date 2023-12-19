import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class apply_gaussian_kernel(nn.Module):
    def __init__(self, imgsize=18):
        super(apply_gaussian_kernel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.beta = beta
        # self.kernel_sigma = kernel_sigma
        #
        # # regular grid / [-1,1] normalized
        # self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
        #                                        np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        # self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).to(device)
        # self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).to(device)
        #
        # # kernels for computing gradients
        # self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,
        #                                                                                                           1,
        #                                                                                                           3).to(
        #     device)
        # self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
        #                                                                                                           3,
        #                                                                                                           1).to(
        #     device)

        # 1-d indices for generating Gaussian kernels
        # self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = np.linspace(0, imgsize - 1, imgsize)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(device)
        # self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = np.linspace(0, imgsize - 1, imgsize)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(device)

        # # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        # self.x_normal = np.linspace(-1, 1, feature_W)
        # self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).to(device)
        # self.y_normal = np.linspace(-1, 1, feature_H)
        # self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).to(device)

    # def apply_gaussian_kernel_detail(self, corr, sigma=5):
    #     b, hw, h, w = corr.size()
    #
    #     idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
    #     idx_y = (idx // w).view(b, 1, 1, h, w).float()
    #     idx_x = (idx % w).view(b, 1, 1, h, w).float()
    #
    #     x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
    #     y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)
    #
    #     gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
    #     gauss_kernel = gauss_kernel.view(b, hw, h, w)
    #
    #     return gauss_kernel * corr
    #
    # def kernel_soft_argmax(self, corr):
    #     corr = self.apply_gaussian_kernel_detail(corr, sigma=self.kernel_sigma)
    #     return corr

    def forward(self, idx_y=0,idx_x=0,imgsize=0,sigma=5):
        # b, hw, h, w = corr.size()

        # idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        # idx_y = (idx // w).view(b, 1, 1, h, w).float()
        # idx_x = (idx % w).view(b, 1, 1, h, w).float()

        # x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        # y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        x = self.x.view(1, 1, imgsize)
        y = self.y.view(1, imgsize, 1)

        # x = self.x.view(1, 1, imgsize).expand(b, 1, imgsize)
        # y = self.y.view(1, imgsize, 1).expand(b, imgsize, 1)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        # gauss_kernel=gauss_kernel/sigma/((2*3.141592653589793)**0.5)
        gauss_kernel = gauss_kernel.reshape(1, imgsize, imgsize)

        return gauss_kernel

class loss_function(nn.Module):
    def __init__(self, args):
        super(loss_function, self).__init__()
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3

        imgsize=288
        # imgsize = 10
        self.apply_gaussian_kernel = apply_gaussian_kernel(imgsize=imgsize)

    def lossfn_two_var(self, target1, target2, num_px = None):
        if num_px is None:
            return torch.sum(torch.pow((target1 - target2),2))
        else:
            return torch.sum(torch.pow((target1 - target2),2) / num_px)
    # def lossfn_two_var(self, target1, target2, num_px = None):
    #     if num_px is None:
    #         return torch.sum(torch.pow((target1 + target2),2))
    #     else:
    #         return torch.sum(torch.pow((target1 + target2),2) / num_px)

    def weight_refine(self, weight,idx_y=None, idx_x=None,new_val=1.0):
        # if idx_x==0:
        #     idx_x_left=idx_x
        #     idx_x_right=idx_x+2
        # elif idx_x==imagesize:
        #     idx_x_left=idx_x-1
        #     idx_x_right = idx_x+1
        # else:
        #     idx_x_left = idx_x - 1
        #     idx_x_right = idx_x+2
        #
        # if idx_y == 0:
        #     idx_y_left = idx_y
        #     idx_y_right=idx_y+2
        # elif idx_y==imagesize:
        #     idx_y_left = idx_y-1
        #     idx_y_right=idx_y+1
        # else:
        #     idx_y_left = idx_y - 1
        #     idx_y_right = idx_y+2
        # print(idx_y_left,idx_y_right,idx_x_left,idx_x_right)
        # print('new_val',new_val)
        # weight[:, idx_y_left:idx_y_right, idx_x_left:idx_x_right] = new_val
        weight[:, idx_y-1:idx_y+2, idx_x-1:idx_x+2] = new_val
        return weight

    def KL_divergence(self,prob, target, weight=None):

        if weight is not None:
            loss = weight * (target * (torch.log(target))  - target * torch.log(prob))
            loss = torch.sum(loss)
        else:
            loss = target * torch.log(target) - target * torch.log(prob)
            loss = torch.sum(loss)
        return loss



    def forward(self,output, image1_points, image2_points, tgt_image_H, tgt_image_W, src_image_H, src_image_W,output_cache=None):


        print('tgt_image_H',tgt_image_H)
        print('tgt_image_W', tgt_image_W)
        print('src_image_H', src_image_H)
        print('src_image_W', src_image_W)

        featmapsize=18
        imgsize=320-16*2
        # keypoints on source and target images
        #source keypoints. Adjust their values.
        source_x = image1_points[0, :]
        source_y = image1_points[1, :]
        print('source_x',source_x)
        print('source_y', source_y)
        # source_x=source_x/src_image_W*(320-16*2)
        # source_y = source_y / src_image_H * (320 - 16 * 2)
        source_x = source_x / src_image_W * featmapsize
        source_y = source_y / src_image_H * featmapsize
        source_x=source_x.long()
        source_y=source_y.long()
        print('source_x', source_x)
        print('source_y', source_y)
        # target keypoints
        target_x = image2_points[0, :]
        target_y = image2_points[1, :]
        print('target_x', target_x)
        print('target_y', target_y)
        target_x_weight = target_x / tgt_image_W * imgsize
        target_y_weight = target_y / tgt_image_H * imgsize
        target_x_weight = target_x_weight.round().long()
        target_y_weight = target_y_weight.round().long()
        target_x = target_x / tgt_image_W * featmapsize
        target_y = target_y / tgt_image_H * featmapsize
        target_x = target_x.long()
        target_y = target_y.long()
        print('target_x', target_x)
        print('target_y', target_y)

        # #estimated probability maps
        # est_target_prob_maps=output['corr_S2T_softmax'][:,:,source_y,source_x]
        # print('est_target_prob_maps.shape',est_target_prob_maps.shape)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ce_loss=0
        ce_loss=torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(device)
        loss_soft=torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(device)
        print('ce_loss',ce_loss)
        print('loss_soft', loss_soft)

        # loss_type = "cross_entropy"
        loss_type = "no_cross_entropy"
  
        #add soft loss but failed
        if loss_type=="binary_cross_entropy":
            for j in range(source_x.size(0)):
                point_x = source_x[j]
                point_y = source_y[j]

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == featmapsize:
                    point_x = point_x - 1

                if point_y == featmapsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)
                est_target_prob_maps = output['corr_S2T_softmax'][:, :, point_y, point_x]
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.max()', est_target_prob_maps.max())
                print('est_target_prob_maps.min()', est_target_prob_maps.min())
                est_target_prob_maps = est_target_prob_maps.reshape(-1, featmapsize, featmapsize).unsqueeze(1)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                est_target_prob_maps = F.interpolate(est_target_prob_maps, size=(imgsize, imgsize), mode='bilinear',
                                                     align_corners=True)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                print('est_target_prob_maps_inte.max()', est_target_prob_maps.max())
                print('est_target_prob_maps_inte.min()', est_target_prob_maps.min())
                # est_target_prob_maps = F.softmax(est_target_prob_maps.reshape(-1, 1, imgsize * imgsize), 2)
                # print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                # print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                # print('est_target_prob_maps_soft.max()', est_target_prob_maps.max())
                # print('est_target_prob_maps_soft.min()', est_target_prob_maps.min())
                est_target_prob_maps = est_target_prob_maps.squeeze(1).reshape(-1, imgsize, imgsize)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)

                point_x = target_x[j]
                point_y = target_y[j]

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == featmapsize:
                    point_x = point_x - 1

                if point_y == featmapsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)

                real_target_prob_maps = output_cache['corr_S2T_softmax'][:, :, point_y, point_x]
                print('real_target_prob_maps.shape', real_target_prob_maps.shape)
                print('real_target_prob_maps.max()', real_target_prob_maps.max())
                print('real_target_prob_maps.min()', real_target_prob_maps.min())
                real_target_prob_maps = real_target_prob_maps.reshape(-1, featmapsize, featmapsize).unsqueeze(1)
                print('real_target_prob_maps.shape', real_target_prob_maps.shape)
                print('real_target_prob_maps.sum()', real_target_prob_maps.sum())
                real_target_prob_maps = F.interpolate(real_target_prob_maps, size=(imgsize, imgsize), mode='bilinear',
                                                     align_corners=True)
                print('real_target_prob_maps_inte.shape', real_target_prob_maps.shape)
                print('real_target_prob_maps_inte.sum()', real_target_prob_maps.sum())
                print('real_target_prob_maps_inte.max()', real_target_prob_maps.max())
                print('real_target_prob_maps_inte.min()', real_target_prob_maps.min())
                # est_target_prob_maps = F.softmax(est_target_prob_maps.reshape(-1, 1, imgsize * imgsize), 2)
                # print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                # print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                # print('est_target_prob_maps_soft.max()', est_target_prob_maps.max())
                # print('est_target_prob_maps_soft.min()', est_target_prob_maps.min())
                real_target_prob_maps = real_target_prob_maps.squeeze(1).reshape(-1, imgsize, imgsize)
                print('real_target_prob_maps.shape', real_target_prob_maps.shape)

                point_x = target_x_weight[j]
                point_y = target_y_weight[j]

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == imgsize:
                    point_x = point_x - 1

                if point_y == imgsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)
                weight = torch.ones(real_target_prob_maps.size()).cuda()
                print('weight.shape', weight.shape)
                weight = self.weight_refine(weight, idx_y=point_y, idx_x=point_x, new_val=(288 * 288 - 3 * 3) / (3 * 3))

                real_target_prob_maps = real_target_prob_maps*self.apply_gaussian_kernel(idx_y=point_y, idx_x=point_x, imgsize=imgsize,sigma=0.6)


                # weight=torch.ones(real_target_prob_maps.size()).cuda()
                # print('weight.shape',weight.shape)
                # weight=self.weight_refine(weight,idx_y=point_y, idx_x=point_x,new_val=(288*288-3*3)/(3*3))
                print('real_target_prob_maps.sum()', real_target_prob_maps.sum())
                print('real_target_prob_maps.max()', real_target_prob_maps.max())
                print('real_target_prob_maps.min()', real_target_prob_maps.min())
                print('ce_loss_each',F.binary_cross_entropy(est_target_prob_maps, real_target_prob_maps, weight=weight,
                                                  size_average=None, reduce=None, reduction='sum'))

                ce_loss += F.binary_cross_entropy(est_target_prob_maps, real_target_prob_maps, weight=weight,
                                                  size_average=None, reduce=None, reduction='sum')
                # ce_loss += F.binary_cross_entropy(real_target_prob_maps, real_target_prob_maps,weight=None, size_average=None, reduce=None, reduction='sum')
                # ce_loss += F.binary_cross_entropy(est_target_prob_maps, est_target_prob_maps, weight=None,
                #                                   size_average=None, reduce=None, reduction='sum')
                # print('ce_loss',ce_loss)
                print('ce_loss', ce_loss)
                # os._exit()

        #Split into two parts
        if loss_type=="kl":
            for j in range(source_x.size(0)):
                point_x = source_x[j]
                point_y = source_y[j]

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == featmapsize:
                    point_x = point_x - 1

                if point_y == featmapsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)
                est_target_prob_maps = output['corr_S2T_softmax'][:, :, point_y, point_x]
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.max()', est_target_prob_maps.max())
                print('est_target_prob_maps.min()', est_target_prob_maps.min())
                est_target_prob_maps = est_target_prob_maps.reshape(-1, featmapsize, featmapsize).unsqueeze(1)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                est_target_prob_maps = F.interpolate(est_target_prob_maps, size=(imgsize, imgsize), mode='bilinear',
                                                     align_corners=True)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                print('est_target_prob_maps_inte.max()', est_target_prob_maps.max())
                print('est_target_prob_maps_inte.min()', est_target_prob_maps.min())
                est_target_prob_maps = F.softmax(est_target_prob_maps.reshape(-1, 1, imgsize * imgsize), 2)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps_soft.sum()', est_target_prob_maps.sum())
                print('est_target_prob_maps_soft.max()', est_target_prob_maps.max())
                print('est_target_prob_maps_soft.min()', est_target_prob_maps.min())
                # est_target_prob_maps = F.softmax(est_target_prob_maps, 2)
                # print('est_target_prob_maps_soft2.sum()', est_target_prob_maps.sum())
                # print('est_target_prob_maps_soft2.max()', est_target_prob_maps.max())
                # print('est_target_prob_maps_soft2.min()', est_target_prob_maps.min())
                est_target_prob_maps = est_target_prob_maps.squeeze(1).reshape(-1, imgsize,imgsize)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)

                point_x = target_x[j]
                point_y = target_y[j]

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == imgsize:
                    point_x = point_x - 1

                if point_y == imgsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)
                real_target_prob_maps = self.apply_gaussian_kernel(idx_y=point_y, idx_x=point_x, imgsize=imgsize,
                                                                   sigma=0.6)

                weight = torch.ones(real_target_prob_maps.size()).cuda()
                #mask
                mask=weight>0
                mask[:, point_y-1:point_y+2, point_x-1:point_x+2]=False


                # print('mask',mask)
                # print('torch.sum(mask)',torch.sum(mask))
                # print('torch.sum(~mask)', torch.sum(~mask))
                # os._exit()

                # print('weight.shape', weight.shape)
                # weight = self.weight_refine(weight, idx_y=point_y, idx_x=point_x, new_val=(288 * 288 - 3 * 3) / (3 * 3)).reshape(-1,imgsize * imgsize)
                print('real_target_prob_maps.sum()', real_target_prob_maps.sum())
                print('real_target_prob_maps.max()', real_target_prob_maps.max())
                print('real_target_prob_maps.min()', real_target_prob_maps.min())
                real_target_prob_maps = F.softmax(real_target_prob_maps.reshape(-1, imgsize * imgsize), 1)
                print('real_target_prob_maps_softmax.shape', real_target_prob_maps.shape)
                print('real_target_prob_maps_softmax.sum()', real_target_prob_maps.sum())
                print('real_target_prob_maps_softmax.max()', real_target_prob_maps.max())
                print('real_target_prob_maps_softmax.min()', real_target_prob_maps.min())
                real_target_prob_maps=real_target_prob_maps.reshape(-1, imgsize , imgsize)
                print('real_target_prob_maps.shape', real_target_prob_maps.shape)


                est_surround=torch.masked_select(est_target_prob_maps,mask)
                est_centor=torch.masked_select(est_target_prob_maps,~mask)
                real_surround=torch.masked_select(real_target_prob_maps,mask)
                real_centor=torch.masked_select(real_target_prob_maps,~mask)
                print('est_surround', est_surround)
                print('real_surround', real_surround)
                print('est_centor', est_centor)
                print('real_centor', real_centor)
                est_surround = est_surround/torch.sum(est_surround)
                est_centor = est_centor/torch.sum(est_centor)
                real_surround = real_surround/torch.sum(real_surround)
                real_centor = real_centor/torch.sum(real_centor)
                print('est_surround', est_surround)
                print('real_surround', real_surround)
                print('est_centor', est_centor)
                print('real_centor', real_centor)

                ce_loss_centor=F.kl_div(est_centor.log(), real_centor,reduction='sum')
                ce_loss_surround = F.kl_div(est_surround.log(), real_surround, reduction='sum')
                print('ce_loss_centor', ce_loss_centor)
                print('ce_loss_surround', ce_loss_surround)
                ce_loss_centor = self.KL_divergence(est_centor, real_centor)
                ce_loss_surround = self.KL_divergence(est_surround, real_surround)
                print('ce_loss_centor',ce_loss_centor)
                print('ce_loss_surround',ce_loss_surround)
                print('torch.sum(est_surround)',torch.sum(est_surround))
                print('torch.sum(real_surround)', torch.sum(real_surround))
                print('torch.sum(est_centor)', torch.sum(est_centor))
                print('torch.sum(real_centor)', torch.sum(real_centor))


                # # ce_loss_centor = F.kl_div(est_centor.log(), real_centor, reduction='sum')
                # #
                # # print('ce_loss_centor', ce_loss_centor)
                # #
                # # ce_loss_centor = self.KL_divergence(est_centor, real_centor)
                # #
                # # print('ce_loss_centor', ce_loss_centor)
                #
                # # # ce_loss += F.binary_cross_entropy(est_target_prob_maps, real_target_prob_maps, weight=None,
                # # #                                   size_average=None, reduce=None, reduction='mean')
                # # ce_loss += F.kl_div(est_target_prob_maps.log(), real_target_prob_maps, reduction='sum')
                # # # print('ce_loss',ce_loss)
                # # # os._exit()
                # #
                # # torchloss = F.kl_div(est_target_prob_maps.log(), real_target_prob_maps,
                # #                      reduction='sum')
                # # myloss = self.KL_divergence(est_target_prob_maps,real_target_prob_maps)
                # # print('torchloss',torchloss)
                # # print('myloss',myloss)
                # print('ce_loss_each', self.KL_divergence(est_target_prob_maps, real_target_prob_maps,weight=weight))
                #
                # ce_loss +=self.KL_divergence(est_target_prob_maps, real_target_prob_maps,weight=weight)

                print('ce_loss_each', ce_loss_centor+100.0*ce_loss_surround)
                ce_loss+=(ce_loss_centor+100.0*ce_loss_surround)


        ## patch level+ soft loss
        if loss_type == "cross_entropy":
            for j in range(source_x.size(0)):
                point_x = source_x[j]
                point_y = source_y[j]

                if point_x == -1 and point_y == -1:
                    continue

                if point_x == featmapsize:
                    point_x = point_x - 1

                if point_y == featmapsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)
                est_target_prob_maps = output['corr_S2T_softmax'][:, :, point_y, point_x]
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.max()', est_target_prob_maps.max())
                print('est_target_prob_maps.min()', est_target_prob_maps.min())
                # est_target_prob_maps = est_target_prob_maps.reshape(-1, featmapsize, featmapsize).unsqueeze(1)
                # print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                # print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                # est_target_prob_maps = F.interpolate(est_target_prob_maps, size=(imgsize, imgsize), mode='bilinear',
                #                                      align_corners=True)
                # print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                # print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                # print('est_target_prob_maps_inte.max()', est_target_prob_maps.max())
                # print('est_target_prob_maps_inte.min()', est_target_prob_maps.min())
                # # est_target_prob_maps = F.softmax(est_target_prob_maps.reshape(-1, 1, imgsize * imgsize), 2)
                # # print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                # # print('est_target_prob_maps.sum()', est_target_prob_maps.sum())
                # # print('est_target_prob_maps_soft.max()', est_target_prob_maps.max())
                # # print('est_target_prob_maps_soft.min()', est_target_prob_maps.min())
                # est_target_prob_maps = est_target_prob_maps.squeeze(1).reshape(-1, imgsize*imgsize)
                est_target_prob_maps = est_target_prob_maps.reshape(-1, featmapsize * featmapsize)
                print('est_target_prob_maps.shape', est_target_prob_maps.shape)
                print('est_target_prob_maps.sum()', est_target_prob_maps.sum())

                point_x = target_x[j]
                point_y = target_y[j]

                if point_x == -1 and point_y == -1:
                    continue

                # if index required, it needs reduction
                if point_x == featmapsize:
                    point_x = point_x - 1

                if point_y == featmapsize:
                    point_y = point_y - 1

                print('point_y', point_y)
                print('point_x', point_x)
                real_target_label = torch.LongTensor([int(point_y) * featmapsize + int(point_x)]).cuda()
                print('real_target_label', real_target_label)
                print('real_target_label.shape', real_target_label.shape)
                ce_loss += F.cross_entropy(est_target_prob_maps, real_target_label)
                print('ce_loss', ce_loss)
                # os._exit()

                if output_cache is not None:
                    point_x = target_x[j]
                    point_y = target_y[j]

                    if point_x == -1 and point_y == -1:
                        continue

                    if point_x == featmapsize:
                        point_x = point_x - 1

                    if point_y == featmapsize:
                        point_y = point_y - 1

                    print('point_y', point_y)
                    print('point_x', point_x)

                    guidance_target_prob_maps = output_cache['corr_S2T_softmax'][:, :, point_y, point_x]
                    print('guidance_target_prob_maps.shape', guidance_target_prob_maps.shape)
                    print('guidance_target_prob_maps.max()', guidance_target_prob_maps.max())
                    print('guidance_target_prob_maps.min()', guidance_target_prob_maps.min())
                    guidance_target_prob_maps = guidance_target_prob_maps.reshape(-1, featmapsize * featmapsize)
                    print('guidance_target_prob_maps.shape', guidance_target_prob_maps.shape)
                    print('guidance_target_prob_maps.sum()', guidance_target_prob_maps.sum())

                    # select top-k, where k=9
                    k = 9
                    guidance_topk_values, guidance_topk_indices = torch.topk(guidance_target_prob_maps, k, dim=1,
                                                                             largest=True, sorted=True, out=None)
                    print('guidance_topk_values', guidance_topk_values)
                    print('guidance_topk_indices', guidance_topk_indices)

                    est_topk_values = torch.index_select(est_target_prob_maps, 1, guidance_topk_indices.squeeze(0))
                    print('est_target_prob_maps', est_target_prob_maps)
                    print('est_topk_values', est_topk_values)

                    tao = 0.5
                    guidance_topk_values_softmax = F.softmax(guidance_topk_values / tao, dim=1)
                    est_topk_values_softmax = F.softmax(est_topk_values, dim=1)

                    loss_soft += (- guidance_topk_values_softmax * (est_topk_values_softmax.log())).sum()
                    print('loss_soft', loss_soft)

                # os._exit()






        print('source_x.size(0)',source_x.size(0))
        ce_loss=ce_loss/source_x.size(0)
        print('ce_loss', ce_loss)

        loss_soft = loss_soft / source_x.size(0)
        print('loss_soft', loss_soft)
        # os_exit

        # 0_1: change affine grid size and value
        big_grid = output['grid_T2S_1'][:, 1:-1, 1:-1, :]
        big_grid[:, :, :, 0] = big_grid[:, :, :, 0] * 10 / 9
        big_grid[:, :, :, 1] = big_grid[:, :, :, 1] * 10 / 9
        big_grid = big_grid.permute(0, 3, 1, 2)  # 1 x 2 x h x w
        tgt_image_H_value = tgt_image_H.int().item()
        tgt_image_W_value = tgt_image_W.int().item()

        grid = F.interpolate(big_grid, size=(tgt_image_H_value, tgt_image_W_value), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # 1 x h x w x 2

        #estimate source points from target points
        point_x = image2_points[0, :].round().long()
        point_y = image2_points[1, :].round().long()
 
        src_image_H_value = src_image_H.int().item()
        src_image_W_value = src_image_W.int().item()
        est_y = ((grid[0, point_y, point_x, 1]+1)*(src_image_H_value-1)/2).unsqueeze(0)
        est_x = ((grid[0, point_y, point_x, 0]+1)*(src_image_W_value-1)/2).unsqueeze(0)
        est_image1_points = torch.cat((est_x, est_y), 0)
        print('happy')

        #calculate loss
        points_dif = torch.pow((image1_points - est_image1_points), 2)
        points_dif_temp = points_dif[0, :] + points_dif[1, :]
        points_dis = points_dif_temp.sqrt()
        points_dis_temp=points_dis.unsqueeze(0)
        aa,bb=points_dis_temp.size()
        L0_1 = torch.sum(points_dis)/bb

        # 0_2: change affine grid size and value
        big_grid = output['grid_T2S_2'][:, 1:-1, 1:-1, :]
        big_grid[:, :, :, 0] = big_grid[:, :, :, 0] * 10 / 9
        big_grid[:, :, :, 1] = big_grid[:, :, :, 1] * 10 / 9
        big_grid = big_grid.permute(0, 3, 1, 2)  # 1 x 2 x h x w
        tgt_image_H_value = tgt_image_H.int().item()
        tgt_image_W_value = tgt_image_W.int().item()

        grid = F.interpolate(big_grid, size=(tgt_image_H_value, tgt_image_W_value), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # 1 x h x w x 2

        #estimate source points from target points
        point_x = image2_points[0, :].round().long()
        point_y = image2_points[1, :].round().long()
 
        src_image_H_value = src_image_H.int().item()
        src_image_W_value = src_image_W.int().item()
        est_y = ((grid[0, point_y, point_x, 1]+1)*(src_image_H_value-1)/2).unsqueeze(0)
        est_x = ((grid[0, point_y, point_x, 0]+1)*(src_image_W_value-1)/2).unsqueeze(0)
        est_image1_points = torch.cat((est_x, est_y), 0)
        print('happy')

        #calculate loss
        points_dif = torch.pow((image1_points - est_image1_points), 2)
        points_dif_temp = points_dif[0, :] + points_dif[1, :]
        points_dis = points_dif_temp.sqrt()
        points_dis_temp=points_dis.unsqueeze(0)
        aa,bb=points_dis_temp.size()
        L0_2 = torch.sum(points_dis)/bb

        # 0_3: change affine grid size and value
        big_grid = output['grid_T2S_3'][:, 1:-1, 1:-1, :]
        big_grid[:, :, :, 0] = big_grid[:, :, :, 0] * 10 / 9
        big_grid[:, :, :, 1] = big_grid[:, :, :, 1] * 10 / 9
        big_grid = big_grid.permute(0, 3, 1, 2)  # 1 x 2 x h x w
        tgt_image_H_value = tgt_image_H.int().item()
        tgt_image_W_value = tgt_image_W.int().item()

        grid = F.interpolate(big_grid, size=(tgt_image_H_value, tgt_image_W_value), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # 1 x h x w x 2

        #estimate source points from target points
        point_x = image2_points[0, :].round().long()
        point_y = image2_points[1, :].round().long()
 
        src_image_H_value = src_image_H.int().item()
        src_image_W_value = src_image_W.int().item()
        est_y = ((grid[0, point_y, point_x, 1]+1)*(src_image_H_value-1)/2).unsqueeze(0)
        est_x = ((grid[0, point_y, point_x, 0]+1)*(src_image_W_value-1)/2).unsqueeze(0)
        est_image1_points = torch.cat((est_x, est_y), 0)
        print('happy')

        #calculate loss
        points_dif = torch.pow((image1_points - est_image1_points), 2)
        points_dif_temp = points_dif[0, :] + points_dif[1, :]
        points_dis = points_dif_temp.sqrt()
        points_dis_temp=points_dis.unsqueeze(0)
        aa,bb=points_dis_temp.size()
        L0_3 = torch.sum(points_dis)/bb

        # 0_4: change affine grid size and value
        big_grid = output['grid_T2S_4'][:, 1:-1, 1:-1, :]
        big_grid[:, :, :, 0] = big_grid[:, :, :, 0] * 10 / 9
        big_grid[:, :, :, 1] = big_grid[:, :, :, 1] * 10 / 9
        big_grid = big_grid.permute(0, 3, 1, 2)  # 1 x 2 x h x w
        tgt_image_H_value = tgt_image_H.int().item()
        tgt_image_W_value = tgt_image_W.int().item()

        grid = F.interpolate(big_grid, size=(tgt_image_H_value, tgt_image_W_value), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # 1 x h x w x 2

        #estimate source points from target points
        point_x = image2_points[0, :].round().long()
        point_y = image2_points[1, :].round().long()
 
        src_image_H_value = src_image_H.int().item()
        src_image_W_value = src_image_W.int().item()
        est_y = ((grid[0, point_y, point_x, 1]+1)*(src_image_H_value-1)/2).unsqueeze(0)
        est_x = ((grid[0, point_y, point_x, 0]+1)*(src_image_W_value-1)/2).unsqueeze(0)
        est_image1_points = torch.cat((est_x, est_y), 0)
        print('happy')

        #calculate loss
        points_dif = torch.pow((image1_points - est_image1_points), 2)
        points_dif_temp = points_dif[0, :] + points_dif[1, :]
        points_dis = points_dif_temp.sqrt()
        points_dis_temp=points_dis.unsqueeze(0)
        aa,bb=points_dis_temp.size()
        L0_4 = torch.sum(points_dis)/bb


        # 1: change affine grid size and value
        # big_grid = output['grid_T2S'][:, 2:-2, 2:-2, :]
        big_grid = output['grid_T2S'][:, 1:-1, 1:-1, :]
        big_grid[:, :, :, 0] = big_grid[:, :, :, 0] * 10 / 9
        big_grid[:, :, :, 1] = big_grid[:, :, :, 1] * 10 / 9
        big_grid = big_grid.permute(0, 3, 1, 2)  # 1 x 2 x h x w
        tgt_image_H_value = tgt_image_H.int().item()
        tgt_image_W_value = tgt_image_W.int().item()
        # tgt_image_H = tgt_image_H.expand_as(big_grid[:, 1, :, :])  # 1 x 2 x h x w
        # tgt_image_W = tgt_image_W.expand_as(big_grid[:, 0, :, :])  # 1 x 2 x h x w
        grid = F.interpolate(big_grid, size=(tgt_image_H_value, tgt_image_W_value), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # 1 x h x w x 2


        #estimate source points from target points
        point_x = image2_points[0, :].round().long()
        point_y = image2_points[1, :].round().long()
        print('point_x',point_x)
        print('point_y',point_y)
        print('grid.size',grid.size())
        src_image_H_value = src_image_H.int().item()
        src_image_W_value = src_image_W.int().item()
        est_y = ((grid[0, point_y, point_x, 1]+1)*(src_image_H_value-1)/2).unsqueeze(0)
        est_x = ((grid[0, point_y, point_x, 0]+1)*(src_image_W_value-1)/2).unsqueeze(0)
        est_image1_points = torch.cat((est_x, est_y), 0)
        print('est_image1_points', est_image1_points)
        print('happy')

        #calculate loss
        points_dif = torch.pow((image1_points - est_image1_points), 2)
        points_dif_temp = points_dif[0, :] + points_dif[1, :]
        points_dis = points_dif_temp.sqrt()
        points_dis_temp=points_dis.unsqueeze(0)
        aa,bb=points_dis_temp.size()
        L1 = torch.sum(points_dis)/bb

        # # calculate flow consistency loss
        # B,C,H,W=output['flow_T2S'].shape
        # L2 = self.lossfn_two_var(output['flow_T2S'], output['warped_flow_T2S'], H*W) # flow consistency  ##flow_T2S is opposite to warped_flow_T2S


        # print('L1',L1)
        # os_exit



        # calculate smoothness loss
        #L3 = torch.sum(output['smoothness_T2S'] )/ 400 # smoothness/grid

        # return self.lambda1 * ce_loss+L1 * self.lambda1, \
        #        L1 * self.lambda1, \
        #        self.lambda1 * ce_loss, \
        #        self.lambda1 * loss_soft
        return L1 * self.lambda1+L0_1 * self.lambda1+L0_2 * self.lambda1+L0_3 * self.lambda1+L0_4 * self.lambda1, \
               L0_1 * self.lambda1, \
               L0_2 * self.lambda1, \
               L0_3 * self.lambda1,\
               L0_4 * self.lambda1,\
               L1 * self.lambda1    




