"""
先运行test获取运行结果，此处运行需要输入模型的输出文件夹
在对应的target_gt中取mask（512），mask resize到256，应用在fake TB上，并且保存下来
target_gt再resize到512，之后就能开始计算SSIM
"""

import os
import cv2
import numpy as np
import re

import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torchvision.transforms import transforms
from sklearn import metrics

from model_eval.kappa import quadratic_weighted_kappa
from options.test_options import TestOptions
# from model_eval.fid_score import get_is_fid_score
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# from model_eval.eval_public import eval_public
# from comparison.cataract_comparison_test_dataset import CataractComparisonTestDataset
from PIL import Image
def model_eval(opt, test_output_dir='test_latest', meters=None, wrap=True, write_res=True):
    # 初始化
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # if opt.load_iter != 0:
    #     result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    # else:
    # in是为了训练时测试，not in是为了直接test时使用的
    if 'result' in test_output_dir:
        result_image_dir = os.path.join(test_output_dir, 'images')
    else:
        result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    # gt_image_dir = os.path.join(opt.dataroot, 'gt')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'gt')
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'sustech')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    if 'pix2pix' in opt.model or 'cycle' in opt.model or 'HFC2stage' == opt.model:
        end_word = 'fake_TB.png'
    elif 'SGRIF' in opt.name:
        end_word = '.jpg'
    else:
        if 'SDA_Source' in opt.model:
            end_word = 'fake_EyeQ.png'
        else:
            end_word = 's_fake.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        # 不是pred的图像不进行评估
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        count += 1


        # gt_image_name = image_num + 'B_reg.jpg'
        if 'DRIVE_simulated' in opt.dataroot:
            gt_image_name = image_num + 'B.png'
        elif 'fiq' in opt.dataroot:
            gt_image_name = image_num + '.png'
        elif 'drive' in opt.dataroot:
            # 处理Drive数据集
            image_name = image_name.split('_')[0] + '_' + image_name.split('_')[1]
            gt_image_name = image_name + '.png'
            image_name = image_name + '_fake_EyeQ.png'
        elif 'EyeQ' in opt.dataroot:
            # 处理EyeQ数据集
            image_num = re.findall(r'[0-9]+', image_name)[0]  # 获取图像的编号
            image_rl = image_name.split('_')[1]  # 获取图像的RL
            gt_image_name = image_num +'_' + image_rl+ '.png'
        # elif 'A2A' in opt.dataroot:
        #
        elif 'RCF' in opt.dataroot:
            if len(image_name.split('_')) == 1:
                image_name = image_name
                gt_image_name = os.path.split(image_name)[-1].split('_')[0].replace('A.jpg', 'B.jpg')
            else:
                image_name = image_name
                gt_image_name = os.path.split(image_name)[-1].split('_')[0].replace('A', 'B.jpg')
            mask_name = image_name.replace('_fake_EyeQ', '')
        else:
            image_name = image_name
            gt_image_name = image_name.split('_')[0]+'.PNG'
        # gt_image_name = image_num + '.png'

        # results/中找pred
        image_path = os.path.join(result_image_dir, image_name)
        # mask_path = os.path.join(opt.dataroot,'mask', mask_name)
        # datasets找gt
        # gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        if 'EyeQ'  in opt.dataroot  or 'drive' in opt.dataroot:
            gt_image_path = os.path.join(opt.dataroot + gt_image_dir, gt_image_name)
        # print(gt_image_path)
        # print(image_path)
        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            gt_image = cv2.imread(gt_image_path.replace('jpg', 'png'))




            # drive数据集的gt图像是png格式的
            if 'drive' in opt.dataroot:
                gt_image = cv2.imread(gt_image_path.replace('png', 'jpg'))
                # 1024*512的图像 取gt
                gt_image = gt_image[:,512:1024]


        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)

        # 取mask
        image_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
        gray = np.array(image_gray)
        threshold = 5
        if '10' in image_name:
            threshold = 10
        mask = ndimage.binary_opening(gray>threshold, structure=np.ones((8,8)))

        # resize gt和mask到256，并应用到image中
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, image_size)
        mask = mask[:, :, np.newaxis]
        gt_image = cv2.resize(gt_image, image_size)
        mask_image = image * mask

        # # TODO:不需要的时候可以注释
        # # 保存mask_image
        # # output_image_name = image_num + '_' + image_name.split('_')[2] + '_' + 'B_reg.jpg'
        temp_code = image_name.split('_')[0]
        cv2.imwrite(os.path.join(post_output_dir, image_name), mask_image)

        # if count == 10 or count % 100 == 0:
        #     print(count)
        # # 分块进行评价
        # if opt.crop_size > 256:
        #     ssim = psnr = 0
        #     for i in range(int(opt.crop_size / 256)):
        #         for j in range(int(opt.crop_size / 256)):
        #             part_of_mask_image = mask_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             part_of_gt_image = gt_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             ssim += structural_similarity(part_of_mask_image, part_of_gt_image, data_range=255, multichannel=True)
        #             psnr += peak_signal_noise_ratio(part_of_gt_image, part_of_mask_image, data_range=255)
        #     ssim /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        #     psnr /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        # else:
        #     # -------------评价代码-------------
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        # psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        if not dict_sum_ssim.get(temp_code):
            dict_sum_ssim[temp_code] = 0
        if not dict_sum_max.get(temp_code):
            dict_sum_max[temp_code] = (0, 0)
        if dict_sum_max[temp_code][1] < ssim:
            dict_sum_max[temp_code] = (image_name, ssim)
        # dict_sum_ssim[temp_code] += ssim
        dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        sum_psnr += psnr
        # -------------评价代码-------------
    # print(dict_sum_max)
    # print(dict_sum_ssim)


    # if cataract_test_dataset is not None:
    #     device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    #     is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    #     with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #         f.write('%f,%f,%f,' % (is_mean, is_std, fid))
    if write_res:
        # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
        # with open(os.path.join('./results', 'log', opt.name + '.csv'), 'a') as f:
        with open(os.path.join( './log', opt.name + '.csv'), 'a') as f:
            if not wrap:
                f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
                if meters is not None:
                    for name, meter in meters.meters.items():
                        f.write('%.4f,' % meter.global_avg)
            else:
                f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
                if meters is not None:
                    for name, meter in meters.meters.items():
                        f.write('%.4f,' % meter.global_avg)
                f.write('\n')
    print('Number for process ssim and psnr:{}'.format(count))
    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count, sum_psnr / count

def model_eval_ultrasound(opt, test_output_dir='test_latest', meters=None, wrap=True, write_res=True):
    # 初始化
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # if opt.load_iter != 0:
    #     result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    # else:
    # in是为了训练时测试，not in是为了直接test时使用的
    if 'result' in test_output_dir:
        result_image_dir = os.path.join(test_output_dir, 'images')
    else:
        result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    # gt_image_dir = os.path.join(opt.dataroot, 'gt')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'gt')
    post_output_dir = os.path.join(opt.results_dir, opt.name, opt.postname)
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    if 'pix2pix' in opt.model or 'cycle' in opt.model or 'HFC2stage' == opt.model:
        end_word = 'fake_B.png'
    elif 'SGRIF' in opt.name:
        end_word = '.jpg'
    else:
        if 'SDA_Source' in opt.model:
            end_word = 'fake_EyeQ.png'
        else:
            end_word = 's_fake.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        # 不是pred的图像不进行评估
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        count += 1


        # gt_image_name = image_num + 'B_reg.jpg'
        if 'DRIVE_simulated' in opt.dataroot:
            gt_image_name = image_num + 'B.png'
        elif 'fiq' in opt.dataroot:
            gt_image_name = image_num + '.png'
        elif 'drive' in opt.dataroot:
            # 处理Drive数据集
            image_name = image_name.split('_')[0] + '_' + image_name.split('_')[1]
            gt_image_name = image_name + '.png'
            image_name = image_name + '_fake_EyeQ.png'
        elif 'EyeQ' in opt.dataroot:
            # 处理EyeQ数据集
            image_num = re.findall(r'[0-9]+', image_name)[0]  # 获取图像的编号
            image_rl = image_name.split('_')[1]  # 获取图像的RL
            gt_image_name = image_num +'_' + image_rl+ '.png'
        # elif 'A2A' in opt.dataroot:
        #
        elif 'ultrasound' in opt.dataroot or 'SUStecH' in opt.dataroot or 'EHFU' in opt.dataroot:
            image_name = image_name
            gt_image_name = image_name.replace('_fake_B.png', '.png')

        elif 'A2A' in opt.dataroot:
            image_name = image_name
            gt_image_name = image_name.split('_')[0] + '_Averaged Image.tif'
        else:
            image_name = image_name
            gt_image_name = image_name.split('_')[0]+'.PNG'
        # gt_image_name = image_num + '.png'

        # results/中找pred
        image_path = os.path.join(result_image_dir, image_name)
        # datasets找gt
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        if 'EyeQ'  in opt.dataroot  or 'drive' in opt.dataroot:
            gt_image_path = os.path.join(opt.dataroot + gt_image_dir, gt_image_name)
        # 读取图像

        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            gt_image = cv2.imread(gt_image_path.replace('jpg', 'png'))




            # drive数据集的gt图像是png格式的
            if 'drive' in opt.dataroot:
                gt_image = cv2.imread(gt_image_path.replace('png', 'jpg'))
                # 1024*512的图像 取gt
                gt_image = gt_image[:,512:1024]


        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size)




        # post processing
        temp_code = image_name.split('_')[0]
        cv2.imwrite(os.path.join(post_output_dir, image_name.replace('_fake_B.png', '.png')),image)

        # gt_image = cv2.resize(gt_image, image_size)
        # # mask_image = image * mask
        # #     # -------------评价代码-------------
        # ssim = structural_similarity(image, gt_image, data_range=255, multichannel=True)
        # # psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        # psnr = peak_signal_noise_ratio(gt_image, image, data_range=255)
        # if not dict_sum_ssim.get(temp_code):
        #     dict_sum_ssim[temp_code] = 0
        # if not dict_sum_max.get(temp_code):
        #     dict_sum_max[temp_code] = (0, 0)
        # if dict_sum_max[temp_code][1] < ssim:
        #     dict_sum_max[temp_code] = (image_name, ssim)
        # # dict_sum_ssim[temp_code] += ssim
        # dict_sum_ssim[temp_code] += ssim
        # sum_ssim += ssim
        # sum_psnr += psnr
        # -------------评价代码-------------
    # print(dict_sum_max)
    # print(dict_sum_ssim)


    # if cataract_test_dataset is not None:
    #     device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    #     is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    #     with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #         f.write('%f,%f,%f,' % (is_mean, is_std, fid))
    # if write_res:
    #     # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #     with open(os.path.join('./results', 'log', opt.name + '.csv'), 'a') as f:
    #         if not wrap:
    #             f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
    #             if meters is not None:
    #                 for name, meter in meters.meters.items():
    #                     f.write('%.4f,' % meter.global_avg)
    #         else:
    #             f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
    #             if meters is not None:
    #                 for name, meter in meters.meters.items():
    #                     f.write('%.4f,' % meter.global_avg)
    #             f.write('\n')
    print('Number for process ssim and psnr:{}'.format(count))
    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count, sum_psnr / count
def model_eval_ultrasound_mask(opt, test_output_dir='test_latest', meters=None, wrap=True, write_res=True):
    # 初始化
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # if opt.load_iter != 0:
    #     result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    # else:
    # in是为了训练时测试，not in是为了直接test时使用的
    if 'result' in test_output_dir:
        result_image_dir = os.path.join(test_output_dir, 'images')
    else:
        result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    # gt_image_dir = os.path.join(opt.dataroot, 'gt')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'gt')
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'sustech')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    if 'pix2pix' in opt.model or 'cycle' in opt.model or 'HFC2stage' == opt.model:
        end_word = 'fake_TB.png'
    elif 'SGRIF' in opt.name:
        end_word = '.jpg'
    else:
        if 'SDA_Source' in opt.model:
            end_word = 'fake_EyeQ.png'
        else:
            end_word = 's_fake.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        # 不是pred的图像不进行评估
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        count += 1


        # gt_image_name = image_num + 'B_reg.jpg'
        if 'DRIVE_simulated' in opt.dataroot:
            gt_image_name = image_num + 'B.png'
        elif 'fiq' in opt.dataroot:
            gt_image_name = image_num + '.png'
        elif 'drive' in opt.dataroot:
            # 处理Drive数据集
            image_name = image_name.split('_')[0] + '_' + image_name.split('_')[1]
            gt_image_name = image_name + '.png'
            image_name = image_name + '_fake_EyeQ.png'
        elif 'EyeQ' in opt.dataroot:
            # 处理EyeQ数据集
            image_num = re.findall(r'[0-9]+', image_name)[0]  # 获取图像的编号
            image_rl = image_name.split('_')[1]  # 获取图像的RL
            gt_image_name = image_num +'_' + image_rl+ '.png'
        # elif 'A2A' in opt.dataroot:
        #
        elif 'ultrasound' in opt.dataroot:
            image_name = image_name
            gt_image_name = image_name.split('-')[0] + '.png'
        else:
            image_name = image_name
            gt_image_name = image_name.split('_')[0]+'.PNG'
        # gt_image_name = image_num + '.png'

        # results/中找pred
        image_path = os.path.join(result_image_dir, image_name)
        # datasets找gt
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        if 'EyeQ'  in opt.dataroot  or 'drive' in opt.dataroot:
            gt_image_path = os.path.join(opt.dataroot + gt_image_dir, gt_image_name)
        print(gt_image_path)
        print(image_path)

        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            gt_image = cv2.imread(gt_image_path.replace('jpg', 'png'))




            # drive数据集的gt图像是png格式的
            if 'drive' in opt.dataroot:
                gt_image = cv2.imread(gt_image_path.replace('png', 'jpg'))
                # 1024*512的图像 取gt
                gt_image = gt_image[:,512:1024]


        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)

        # 取mask
        # image_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
        # gray = np.array(image_gray)
        # threshold = 5
        # if '10' in image_name:
        #     threshold = 10
        # mask = ndimage.binary_opening(gray>threshold, structure=np.ones((8,8)))
        #
        mask_path = os.path.join(opt.dataroot, 'mask', 'mask.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        mask = mask / 255
        mask = mask.astype(np.uint8)
        mask = mask[:, :, np.newaxis]
        gt_image = cv2.resize(gt_image, image_size)
        mask_image = image * mask
        gt_image = gt_image * mask

        # # TODO:不需要的时候可以注释
        # # 保存mask_image
        # # output_image_name = image_num + '_' + image_name.split('_')[2] + '_' + 'B_reg.jpg'
        temp_code = image_name.split('_')[0]
        cv2.imwrite(os.path.join(post_output_dir, image_name),image)

        if count == 10 or count % 100 == 0:
            print(count)
        # # 分块进行评价
        # if opt.crop_size > 256:
        #     ssim = psnr = 0
        #     for i in range(int(opt.crop_size / 256)):
        #         for j in range(int(opt.crop_size / 256)):
        #             part_of_mask_image = mask_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             part_of_gt_image = gt_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             ssim += structural_similarity(part_of_mask_image, part_of_gt_image, data_range=255, multichannel=True)
        #             psnr += peak_signal_noise_ratio(part_of_gt_image, part_of_mask_image, data_range=255)
        #     ssim /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        #     psnr /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        # else:
        #     # -------------评价代码-------------
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        # psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        if not dict_sum_ssim.get(temp_code):
            dict_sum_ssim[temp_code] = 0
        if not dict_sum_max.get(temp_code):
            dict_sum_max[temp_code] = (0, 0)
        if dict_sum_max[temp_code][1] < ssim:
            dict_sum_max[temp_code] = (image_name, ssim)
        # dict_sum_ssim[temp_code] += ssim
        dict_sum_ssim[temp_code] += ssim
        sum_ssim += ssim
        sum_psnr += psnr
        # -------------评价代码-------------
    # print(dict_sum_max)
    # print(dict_sum_ssim)


    # if cataract_test_dataset is not None:
    #     device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    #     is_mean, is_std, fid = get_is_fid_score(cataract_test_dataset, device=device, dataroot=opt.dataroot)
    #     with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #         f.write('%f,%f,%f,' % (is_mean, is_std, fid))
    # if write_res:
    #     # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #     with open(os.path.join('./results', 'log', opt.name + '.csv'), 'a') as f:
    #         if not wrap:
    #             f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
    #             if meters is not None:
    #                 for name, meter in meters.meters.items():
    #                     f.write('%.4f,' % meter.global_avg)
    #         else:
    #             f.write('%f,%f,' % (sum_ssim / count, sum_psnr / count))
    #             if meters is not None:
    #                 for name, meter in meters.meters.items():
    #                     f.write('%.4f,' % meter.global_avg)
    #             f.write('\n')
    print('Number for process ssim and psnr:{}'.format(count))
    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count, sum_psnr / count
def fiq_evaluation(opt, test_output_dir='test_latest'):
    # 初始化
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # test_output_dir = 'test_latest'
    if opt.load_iter != 0:
        result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_fiq_latest_iter' + str(opt.load_iter) + '/images')
    else:
        # in是为了训练时测试
        if 'result' in test_output_dir:
            result_image_dir = os.path.join(test_output_dir, 'images')
        else:
            result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
        gt_mask_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir.replace('_mask', '') + '_mask')
        input_mask_image_dir = os.path.join(opt.dataroot, 'input_mask')
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'gt')
        gt_mask_image_dir = os.path.join(opt.dataroot, 'target_gt_mask')

    # 可视化
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'sustech')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    # 前期准备
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    # end_word = 'fake_B.png'
    if 'pix2pix' in opt.model or 'cycle' in opt.model:
        end_word = 'fake_TB.png'
    elif 'SGRIF' in opt.name:
        end_word = '.png'
    else:
        if 'SDA_Source' in opt.model:
            end_word = 'fake_EyeQ.png'
        else:
            end_word = 's_fake.png'

    # print(end_word)
    for image_name in image_name_list:

        # TODO:对于cycle应该是fake_B.png
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        image_num = re.findall(r'[0-9]+', image_name)[0]
        gt_image_name = image_num + '.png'
        # if 'avr_test' in opt.target_gt_dir:
        #     gt_image_name = image_name.split('-')[0] + '.png'

        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        mask_path = os.path.join(gt_mask_image_dir, gt_image_name)
        input_path = os.path.join(input_mask_image_dir,gt_image_name)
        # 读取图像
        try:
            gt_image = cv2.imread(gt_image_path)
            if gt_image is None:
                raise Exception('no gt images')
        except:
            continue
        count += 1

        gt_image = cv2.resize(gt_image, image_size)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)

        # 读取mask并进行预处理
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        mask = mask / 255
        mask = mask.astype(np.uint8)
        mask = mask[:, :, np.newaxis]

        # input_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        # input_mask = cv2.resize(input_mask, image_size, interpolation=cv2.INTER_NEAREST)
        # input_mask = input_mask / 255
        # input_mask = input_mask.astype(np.uint8)
        # input_mask = input_mask[:, :, np.newaxis]

        mask_image = image * mask
        gt_image = gt_image * mask


        cv2.imwrite(os.path.join(post_output_dir, image_name), mask_image)

        # if count == 10 or count % 100 == 0:
        #     print(count)

        # 评价
        # if opt.crop_size > 256:
        #     ssim = psnr = 0
        #     for i in range(int(opt.crop_size / 256)):
        #         for j in range(int(opt.crop_size / 256)):
        #             part_of_mask_image = mask_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             part_of_gt_image = gt_image[i*256:(i+1)*256, j*256:(j+1)*256]
        #             ssim += structural_similarity(part_of_mask_image, part_of_gt_image, data_range=255, multichannel=True)
        #             psnr += peak_signal_noise_ratio(part_of_gt_image, part_of_mask_image, data_range=255)
        #     ssim /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        #     psnr /= int(opt.crop_size / 256) * int(opt.crop_size / 256)
        # else:
        #     # -------------评价代码-------------
        #     ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        #     psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)

        sum_ssim += ssim
        sum_psnr += psnr
    print('Number for process ssim and psnr:{}'.format(count))

    print('Test result: ssim: {:.3f}, psnr: {:.2f}'.format(sum_ssim / count, sum_psnr / count))
    return sum_ssim / count, sum_psnr / count


def model_eval_diagnosis(opt, test_output_dir='test_latest',model=None,epoch=None,
                         cat_image_list=None,cat_label_list=None):



    # 初始化
    output_dir = os.path.join(opt.dataroot, "log",opt.name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    losses = []
    model.eval()
    # TODO:确定数据类型
    all_preds = []
    all_labels = []
    all_image_names = []
    all_pred_tensors = []
    model.eval()
    image_size = (256, 256)
    # image_size = (opt.crop_size, opt.crop_size)
    print('evaluating in the images size of {}'.format(image_size))
    # if opt.load_iter != 0:
    #     result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    # else:
    # in是为了训练时测试，not in是为了直接test时使用的
    if 'result' in test_output_dir:
        result_image_dir = os.path.join(test_output_dir, 'images')
    else:
        result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    image_name_list = os.listdir(result_image_dir)
    sum_psnr = sum_ssim = count = 0
    # dict_sum_ssim = {}
    # dict_sum_max = {}

    if 'SDA_Source' in opt.model:
        end_word = 'fake_EyeQ.png'
    else:
        end_word = 's_fake.png'


    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        # 不是pred的图像不进行评估
        if not image_name.endswith(end_word):
            continue
        # if image_name.replace("_"+end_word, '')+".png" not in cat_image_list:
        #     continue
        # 初始化操作
        count += 1

        # results/中找pred
        image_path = os.path.join(result_image_dir, image_name)
        # 读取input并将其转换成tensor
        input = Image.open(image_path).convert('RGB')
        # 将input transform
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        input= transform(input)
        input = input.unsqueeze(0)
        input = input.cuda()
        index = cat_image_list.index(image_name.replace("_"+end_word, '')+".png")
        label = cat_label_list[index]
        label = torch.tensor(label)
        prediction = model(input)
        # ----------获取分类结果，进行投票，统计结果-------------
        preds = torch.argmax(prediction, dim=1).cpu().tolist()
        all_labels += list(list([label]))
        all_preds += preds
        all_image_names += list(list([image_name.replace("_"+end_word, '')+".png"]))
        all_pred_tensors.append(prediction)

    # TODO：保存结果all_labels，all_preds
    all_pred_tensors = torch.cat(all_pred_tensors, dim=0).cpu()
    # gt = torch.eye(cfg.NUM_CLASSES)[all_labels, :].flatten()
    # pr = all_pred_tensors.flatten()
    gt = torch.eye(5)[all_labels, :]
    pr = all_pred_tensors

    res_file_name = os.path.join(output_dir, 'pred_res_{}.csv'.format(epoch))
    with open(res_file_name, 'w') as f:
        for i in range(len(all_image_names)):
            f.write('{},{},{}\n'.format(all_image_names[i], all_preds[i], all_labels[i]))

    loss_avg = np.array(losses).mean()
    acc = accuracy_score(all_labels, all_preds)
    rec_avg = recall_score(all_labels, all_preds, average='weighted')
    prec_avg = precision_score(all_labels, all_preds, average='weighted')
    f1_avg = f1_score(all_labels, all_preds, average='weighted')
    print('F1', f1_score(all_labels, all_preds, average=None))
    print('precision', precision_score(all_labels, all_preds, average=None))
    print('recall', recall_score(all_labels, all_preds, average=None))

    quad_kappa = quadratic_weighted_kappa(all_labels, all_preds)
    cohen_kappa = metrics.cohen_kappa_score(all_labels, all_preds)
    auc = metrics.roc_auc_score(gt, pr)

    pred_eval_name = os.path.join(output_dir, '{}.csv'.format(opt.name))
    with open(pred_eval_name, 'a') as f:
        f.write('{},{},{},{},{},{},{},{}, {}\n'.format(epoch, loss_avg, acc, rec_avg,
                                                       prec_avg, quad_kappa, auc, f1_avg, cohen_kappa))

    # 数据统计/输出
    eval_type_dict = {
        'loss': loss_avg,
        # 'Accuracy': acc,
        # 'Precision': prec_avg,
        # 'Recall': rec_avg,
        'AUC': auc,
        'F1_score': f1_avg,
        # 'QKappa': quad_kappa,
        'Ckappa': cohen_kappa

    }
    print(eval_type_dict)
    # 重新开启训练模式
    model.train()
    return eval_type_dict


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test_total options
    model_eval(opt)
    # cataractTestDataset = CataractTestDataset(opt, test_web_dir)
    # eval_public()