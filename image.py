import torch
import torch.nn.functional as F
import numpy as np
import os
import random
from custom_dataset import PF_Pascal
from model import SFNet
# import matplotlib.pyplot as plt
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description="SFNet evaluation")
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='decaying factor')
parser.add_argument('--decay_schedule', type=str, default='30', help='learning rate decaying schedule')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--feature_h', type=int, default=20, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=20, help='width of feature volume')


parser.add_argument('--train_csv_path', type=str, default='./data/train_pairs_pf_pascal.csv',
                    help='directory of train csv file')
parser.add_argument('--train_image_path', type=str, default='./data/PF_Pascal/', help='directory of train images')
parser.add_argument('--train_mask_path', type=str, default='./data/VOC2012_seg_msk.npy',
                    help='directory of pre-processed(.npy) foreground masks')
parser.add_argument('--valid_csv_path', type=str, default='./data/bbox_test_pairs_pf.csv',
                    help='directory of validation csv file')
parser.add_argument('--valid_image_path', type=str, default='./data/PF-dataset', help='directory of validation data')
parser.add_argument('--test_csv_path', type=str, default='./data/bbox_test_pairs_pf_pascal.csv', help='directory of## test csv file')
parser.add_argument('--test_image_path', type=str, default='./data/PF_Pascal/', help='directory of test data')
parser.add_argument('--beta', type=float, default=100, help='inverse temperature of softmax @ kernel soft argmax')
parser.add_argument('--kernel_sigma', type=float, default=5,
                    help='standard deviation of Gaussian kerenl @ kernel soft argmax')
parser.add_argument('--lambda1', type=float, default=1, help='weight parameter of offset loss')
parser.add_argument('--lambda2', type=float, default=200, help='weight parameter of flow consistency loss')
parser.add_argument('--lambda3', type=float, default=200, help='weight parameter of smoothness loss')
parser.add_argument('--eval_type', type=str, default='image_size', choices=('bounding_box', 'image_size'),
                    help='evaluation type for PCK threshold (bounding box | image size)')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--vis', action='store_true', help='enable visualization')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if not args.vis:
    args.vis=False

vis=args.vis
dir_path='./vision_sa_pascal'
if vis:
    if os.path.exists(dir_path):
        # 如果目录存在，遍历目录中的文件和子目录，并删除它们
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
    else:
        # 如果目录不存在，创建新目录
        os.makedirs(dir_path)

# Set seed
global global_seed
global_seed = args.seed
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
torch.backends.cudnn.deterministic = True

# Data Loader
print("Instantiate dataloader")
test_dataset = PF_Pascal(args.test_csv_path, args.test_image_path, args.feature_h, args.feature_w, args.eval_type)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False, num_workers=args.num_workers)

# Instantiate model
print("Instantiate model")
net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma=args.kernel_sigma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

# Load weights
print("Load pre-trained weights")
best_weights = torch.load("./weights/best_checkpoint_pascal.pt")

net_state = best_weights['net']
net.load_state_dict(net_state, strict=False)



# PCK metric from 'https://github.com/ignacio-rocco/weakalign/blob/master/util/eval_util.py'
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    p_src = source_points[0, :]
    p_wrp = warped_points[0, :]

    N_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :N_pts] - p_wrp[:, :N_pts], 2), 0), 0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck 
    
def tensor2arrary_image(tgt_var_image):
    tgt_var_image_numpy=tgt_var_image.squeeze(0).permute(1,2,0).numpy()#b*c*h*w->h*w*c
    tgt_var_image_numpy=tgt_var_image_numpy*255
    tgt_var_image_numpy_RGB = cv2.cvtColor(tgt_var_image_numpy, cv2.COLOR_BGR2RGB)#bgr->rgb
    return tgt_var_image_numpy_RGB
    
def blockimage(tgt_var_image_numpy_RGB, interval=16):
    tgt_h, tgt_w = tgt_var_image_numpy_RGB.shape[0], tgt_var_image_numpy_RGB.shape[1]
    for x in range(0, tgt_w, interval):
        greeen = (0,255,0)
        cv2.line(tgt_var_image_numpy_RGB,(0,x),(tgt_h,x),greeen,1)  # (h,w),(h,w)
    for y in range(0, tgt_h, interval):
        greeen = (0,255,0)
        cv2.line(tgt_var_image_numpy_RGB,(y,0),(y,tgt_w),greeen,1)
    return tgt_var_image_numpy_RGB
    
#def tgt_croping(tgt_var_image, block_i=10, block_j=5, interval=16):
def tgt_croping(tgt_var_image, block_i=5, block_j=7, interval=16):
    new_img_PIL = transforms.ToPILImage()(tgt_var_image.squeeze(0)).convert('RGB')
    region = new_img_PIL.crop((16*(block_i-1),16*(block_j-1), 16*block_i, 16*block_j))# (left x, up y, right x+w, below y+h)
    region.save("./tgt_focus.jpg")
    region = new_img_PIL.crop((16*(block_i-1), 16*(block_j-2), 16*block_i, 16*(block_j-1)))
    region.save("./tgt_focus_up.jpg")
    region = new_img_PIL.crop((16*(block_i-2), 16*(block_j-1), 16*(block_i-1), 16*block_j))
    region.save("./tgt_focus_left.jpg")
    region = new_img_PIL.crop((16*block_i, 16*(block_j-1), 16*(block_i+1), 16*block_j))
    region.save("./tgt_focus_right.jpg")
    region = new_img_PIL.crop((16*(block_i-1), 16*block_j, 16*block_i, 16*(block_j+1)))
    region.save("./tgt_focus_down.jpg")
    new_img_PIL.save("./tgt_PIL.jpg")
    return new_img_PIL


with torch.no_grad():
    print('Computing PCK@Test set...')
    net.eval()
    total_correct_points01 = 0
    total_correct_points03 = 0
    total_correct_points05 = 0
    total_correct_points10 = 0
    total_correct_points15 = 0
    total_points = 0
    total_smoothness = 0.0
    for i, batch in enumerate(test_loader):
        src_image = batch['image1'].to(device)
        tgt_image = batch['image2'].to(device)
        output = net(src_image, tgt_image, train=False)

        # save original src and tgt
        tgt_var_image = batch['image2_var_rgb']
        tgt_var_image_numpy_RGB = tensor2arrary_image(tgt_var_image)  # tensor to arrary for cv2

        if vis:
            name = "./vision_sa_pascal/" + str(i) + "tgt.jpg"
            cv2.imwrite(name, tgt_var_image_numpy_RGB)
        # else:
        #     cv2.imwrite("tgt.jpg", tgt_var_image_numpy_RGB)
        # tgt_block = blockimage(tgt_var_image_numpy_RGB)  # drawing grids on target image
        # tgt_var_image_PIL_RGB = tgt_croping(tgt_var_image, block_i=5 + 7, block_j=7 - 2,
        #                                     interval=16)  # crop target image around center block

        src_var_image = batch['image1_var_rgb']
        src_var_image_numpy_RGB = tensor2arrary_image(src_var_image)
        if vis:
            name = "./vision_sa_pascal/" + str(i) + "scr.jpg"
            cv2.imwrite(name, src_var_image_numpy_RGB)
        # else:
        #     cv2.imwrite("scr.jpg", src_var_image_numpy_RGB)
        # src_block = blockimage(src_var_image_numpy_RGB)

        # cv2.imwrite("tgt_block.jpg", tgt_block)
        # cv2.imwrite("scr_block.jpg", src_block)

        # # Estimate warped images
        # # using consistency loss can obtain better image warping with pixel-wise mapping
        # grid_T2S = output['grid_T2S'].permute(0, 3, 1, 2)
        # grid_T2S = F.interpolate(grid_T2S, size=(320, 320), mode='bilinear', align_corners=True)
        # grid_T2S = grid_T2S.permute(0, 2, 3, 1)
        # warped_src = F.grid_sample(src_var_image, grid_T2S.cpu(), mode='bilinear')
        # warped_src_numpy = tensor2arrary_image(warped_src)  # tensor to arrary for cv2
        # if vis:
        #     name = "./vision_sa_pascal/" + str(i) + "warped_src.jpg"
        #     cv2.imwrite(name, warped_src_numpy)
        # else:
        #     cv2.imwrite("warped_src.jpg", warped_src_numpy)
        # warped_src_block = blockimage(warped_src_numpy)  # drawing grids on warped source image
        # cv2.imwrite("warped_src_block.jpg", warped_src_block)

        # grid_S2T = output['grid_S2T'].permute(0, 3, 1, 2)
        # grid_S2T = F.interpolate(grid_S2T, size=(320, 320), mode='bilinear', align_corners=True)
        # grid_S2T = grid_S2T.permute(0, 2, 3, 1)
        # warped_tgt = F.grid_sample(tgt_var_image, grid_S2T.cpu(), mode='bilinear')
        # warped_tgt_numpy = tensor2arrary_image(
        #     warped_tgt)  # tensor to arrary for cv2 :b*c*h*w->h*w :bgr->rgb:0-1 ->0-255
        # if vis:
        #     name = "./vision_sa_pascal/" + str(i) + "warped_tgt.jpg"
        #     cv2.imwrite(name, warped_tgt_numpy)
        # else:
        #     cv2.imwrite("warped_tgt.jpg", warped_tgt_numpy)

        # # save flow image flow_T2S
        # flow_T2S = output['flow_T2S']  # b * c * h * w
        # flow_T2S = F.interpolate(flow_T2S, size=(320, 320), mode='bilinear', align_corners=True)
        # flow_b, flow_c, flow_h, flow_w = flow_T2S.size()
        # flow_T2S = flow_T2S.permute(0, 2, 3, 1).reshape(1, flow_h * flow_w, flow_c)  # b * (h * w) * c
        # # print('flow_T2S', flow_T2S)
        # flow_T2S = (flow_T2S + 1) / 2.0
        # flow_T2S_min, _ = flow_T2S.min(dim=1, keepdim=True)  # find min
        # # print('flow_T2S_min', flow_T2S_min)
        # flow_T2S = flow_T2S - flow_T2S_min
        # flow_T2S_max, _ = flow_T2S.max(dim=1, keepdim=True)  # find max
        # # print('flow_T2S_max', flow_T2S_max)
        # flow_T2S = flow_T2S / flow_T2S_max  # x -> (x-min)/max((x-min))
        # flow_T2S = flow_T2S.reshape(1, flow_h, flow_w, flow_c)  # b * h * w * c
        # # print('flow_T2S', flow_T2S)
        # flow_b, flow_h, flow_w, _ = flow_T2S.size()
        # flow_T2S_image = torch.FloatTensor(flow_b, flow_h, flow_w, 3).fill_(0.0)  # create flow_T2S_image for storage
        # flow_T2S_image[:, :, :, 0] = flow_T2S[:, :, :, 0]
        # flow_T2S_image[:, :, :, 2] = flow_T2S[:, :, :, 1]
        # flow_T2S_image = flow_T2S_image * 255.0
        # # print('flow_T2S_image', flow_T2S_image)
        # flow_T2S_image = flow_T2S_image.squeeze(0).cpu().numpy()  # tensor to arrary for cv2 b*h*w*c->h*w*c
        # cv2.imwrite("flow_T2S_image.png", flow_T2S_image)
        # img = cv2.imread('flow_T2S_image.png')
        # # print('img', img)
        # # save flow image flow_S2T
        # flow_S2T = output['flow_S2T']  # b * c * h * w
        # flow_S2T = F.interpolate(flow_S2T, size=(320, 320), mode='bilinear', align_corners=True)
        # flow_b, flow_c, flow_h, flow_w = flow_S2T.size()
        # flow_S2T = flow_S2T.permute(0, 2, 3, 1).reshape(1, flow_h * flow_w, flow_c)  # b * (h * w) * c
        # flow_S2T = (flow_S2T + 1) / 2.0
        # flow_S2T_min, _ = flow_S2T.min(dim=1, keepdim=True)  # find min
        # flow_S2T = flow_S2T - flow_S2T_min
        # flow_S2T_max, _ = flow_S2T.max(dim=1, keepdim=True)  # find max
        # flow_S2T = flow_S2T / flow_S2T_max  # x -> (x-min)/max((x-min))
        # flow_S2T = flow_S2T.reshape(1, flow_h, flow_w, flow_c)  # b * h * w * c
        # flow_b, flow_h, flow_w, _ = flow_S2T.size()
        # flow_S2T_image = torch.FloatTensor(flow_b, flow_h, flow_w, 3).fill_(0.0)  # create flow_T2S_image for storage
        # flow_S2T_image[:, :, :, 0] = flow_S2T[:, :, :, 0]
        # flow_S2T_image[:, :, :, 2] = flow_S2T[:, :, :, 1]
        # flow_S2T_image = flow_S2T_image * 255.0
        # flow_S2T_image = flow_S2T_image.squeeze(0).cpu().numpy()  # tensor to arrary for cv2 b*h*w*c->h*w*c
        # cv2.imwrite("flow_S2T_image.png", flow_S2T_image)

        small_grid = output['grid_T2S'][:, 1:-1, 1:-1, :]
        small_grid[:, :, :, 0] = small_grid[:, :, :, 0] * (args.feature_w // 2) / (args.feature_w // 2 - 1)
        small_grid[:, :, :, 1] = small_grid[:, :, :, 1] * (args.feature_h // 2) / (args.feature_h // 2 - 1)
        src_image_H = int(batch['image1_size'][0][0])
        src_image_W = int(batch['image1_size'][0][1])
        tgt_image_H = int(batch['image2_size'][0][0])
        tgt_image_W = int(batch['image2_size'][0][1])
        small_grid = small_grid.permute(0, 3, 1, 2)
        grid = F.interpolate(small_grid, size=(tgt_image_H, tgt_image_W), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)
        grid_np = grid.cpu().data.numpy()

        image1_points = batch['image1_points'][0]
        image2_points = batch['image2_points'][0]

        # smoothness metric   average flow gradient(horizontal and vertical direction) for each flow map
        smoothness_T2S = output['smoothness_T2S'][:, :, 1:-1, 1:-1]  # b * c * feature_h-2 * feature_w-2
        smoothness_T2S = F.interpolate(smoothness_T2S, size=((args.feature_h - 2) * 16, (args.feature_w - 2) * 16),
                                       mode='bilinear',
                                       align_corners=True)  # b * c * ((feature_h-2)*16) * ((feature_w-2)*16)
        smoothness_b, smoothness_c, smoothness_h, smoothness_w = smoothness_T2S.size()
        total_smoothness += (torch.sum(smoothness_T2S) * args.feature_h * 16 / 2 / (
                smoothness_h * smoothness_w))  # accumulate average flow gradient of each flow map

        est_image1_points = np.zeros((2, image1_points.size(1)))
        for j in range(image2_points.size(1)):
            point_x = int(np.round(image2_points[0, j]))
            point_y = int(np.round(image2_points[1, j]))

            if point_x == -1 and point_y == -1:
                continue

            if point_x == tgt_image_W:
                point_x = point_x - 1

            if point_y == tgt_image_H:
                point_y = point_y - 1

            est_y = (grid_np[0, point_y, point_x, 1] + 1) * (src_image_H - 1) / 2
            est_x = (grid_np[0, point_y, point_x, 0] + 1) * (src_image_W - 1) / 2
            est_image1_points[:, j] = [est_x, est_y]

        total_correct_points01 += correct_keypoints(batch['image1_points'],
                                                    torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'],
                                                    alpha=0.01)

        total_correct_points03 += correct_keypoints(batch['image1_points'],
                                                    torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'],
                                                    alpha=0.03)

        total_correct_points05 += correct_keypoints(batch['image1_points'],
                                                  torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'],
                                                  alpha=0.05)
        total_correct_points10 += correct_keypoints(batch['image1_points'],
                                                  torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'],
                                                  alpha=0.10)
        total_correct_points15 += correct_keypoints(batch['image1_points'],
                                                  torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'],
                                                  alpha=0.15)

        # draw ground truth and estimation
        print('gt_keypoints', batch['image1_points'])
        print('est_keypoints', torch.FloatTensor(est_image1_points).unsqueeze(0))

        for j in range(batch['image1_points'].size(2)):
            r_rand = float(np.random.rand(1))
            g_rand = float(np.random.rand(1))
            b_rand = float(np.random.rand(1))

            point_x = int(np.round(batch['image1_points'][0, 0, j]) / src_image_W * (320 - 20 - 20) + 20)
            point_y = int(np.round(batch['image1_points'][0, 1, j]) / src_image_H * (320 - 20 - 20) + 20)

            cv2.circle(src_var_image_numpy_RGB, (point_x, point_y), 4,
                       (np.round(255 * b_rand), np.round(255 * g_rand), np.round(255 * b_rand)), -1)

            # cv2.circle(src_var_image_numpy_RGB, (point_x, point_y), 4, (255/(j+1), 10*j, exp(j)*255/exp(batch['image1_points'].size(2))), -1)

            point_x_es = int(np.round(torch.FloatTensor(est_image1_points)[0, j]) / src_image_W * (320 - 20 - 20) + 20)
            point_y_es = int(np.round(torch.FloatTensor(est_image1_points)[1, j]) / src_image_H * (320 - 20 - 20) + 20)

            cv2.rectangle(src_var_image_numpy_RGB, (point_x_es - 5, point_y_es - 5), (point_x_es + 5, point_y_es + 5),
                          (np.round(255 * b_rand), np.round(255 * g_rand), np.round(255 * b_rand)), 2)

            cv2.line(src_var_image_numpy_RGB, (point_x, point_y), (point_x_es, point_y_es),
                     (np.round(255 * b_rand), np.round(255 * g_rand), np.round(255 * b_rand)), 2)
            # cv2.circle(src_var_image_numpy_RGB, (point_x, point_y), 4, (10*j, 5*j, 255), -1)

        if vis:
            name = "./vision_sa_pascal/" + str(i) + "scr_keypoints.jpg"
            cv2.imwrite(name, src_var_image_numpy_RGB)
        # else:
        #     cv2.imwrite("scr_keypoints.jpg", src_var_image_numpy_RGB)
        # cv2.imwrite("scr_keypoints.png", src_var_image_numpy_RGB)
        # cv2.imwrite("tgt_keypoint.jpg", tgt_var_image_numpy_RGB)
        # os._exit()

    PCK01 = total_correct_points01 / len(test_dataset)
    print('PCK01: %5f' % PCK01)
    PCK03 = total_correct_points03 / len(test_dataset)
    print('PCK03: %5f' % PCK03)
    PCK05 = total_correct_points05 / len(test_dataset)
    print('PCK05: %5f' % PCK05)
    PCK10 = total_correct_points10 / len(test_dataset)
    print('PCK10: %5f' % PCK10)
    PCK15 = total_correct_points15 / len(test_dataset)
    print('PCK15: %5f' % PCK15)
    smoothness = total_smoothness / len(test_dataset)
    print('smoothness: %5f' % smoothness)
