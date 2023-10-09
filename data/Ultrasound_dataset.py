import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class UltrasoundDataset(BaseDataset):
    """A data class for paired images data.

    It assumes that the directory '/path/to/images/train' contains images pairs in the form of {A,B}.
    During test_total time, you need to prepare a directory '/path/to/images/test_total'.
    """

    def __init__(self, opt):
        """Initialize this data class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_source_noise = os.path.join(opt.dataroot, 'input')  # get the images directory
        # self.dir_source_gt = os.path.join(opt.dataroot, 'high_quality')  # get the images directory
        self.dir_source_noise = os.path.join(opt.dataroot, 'all_low')  # get the images directory
        self.dir_source_gt = os.path.join(opt.dataroot, 'all_high')  # get the images directory
        # self.dir_source_noise = os.path.join(opt.dataroot, 'test')  # get the images directory
        # self.dir_source_gt = os.path.join(opt.dataroot, 'test')  # get the images directory


        # self.source_noise_paths = sorted(make_dataset(self.dir_source_noise, opt.max_dataset_size))  # get images paths
        # self.source_gt_paths = sorted(make_dataset(self.dir_source_gt, opt.max_dataset_size))  # get images paths
        # self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get images paths
        # self.input_mask_paths = sorted(make_dataset(self.input_mask, opt.max_dataset_size))  # get images paths
        self.A_path = sorted(make_dataset(self.dir_source_noise, opt.max_dataset_size))  # get images paths
        self.B_path = sorted(make_dataset(self.dir_source_gt, opt.max_dataset_size))
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded images

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain# get images paths
        self.model = opt.model
        self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1))


    def __getitem__(self, index):

        image_path = self.A_path[index]
        image_name = os.path.split(image_path)[-1]
        # 为了适配target
        # image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'
        # gt_path = self.image_paths[random.randint(0, len(self.image_paths) - 1)].split('-')[0].replace('.png',
        #                                                                                                '').replace(
        #     'source_image', 'source_gt') + '.png'

        gt_path = os.path.join(self.dir_source_gt, image_name)
        A = Image.open(image_path).convert('L')
        B = Image.open(gt_path).convert('L')

        # w, h = A.size
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        transform_params = get_params(self.opt, A.size)


        source_noise = self.transform_A(A)
        source_gt = self.transform_A(B)


        return {'A': source_noise, 'B': source_gt, 'A_paths': self.A_path[index], 'B_paths': self.B_path[index]}


    def __len__(self):
        """Return the total number of images in the data."""
        return len(self.A_path)
