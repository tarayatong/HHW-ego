import os
import os.path as osp
import cv2
import math
import random
import time
import numpy as np
import scipy.io as sio
import torch
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
# import h5py

@DATASET_REGISTRY.register()
class TeleHyperDataset(data.Dataset):
    def __init__(self, opt):
        super(TeleHyperDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt =  {'type': 'disk'}
        self.gt_folder = opt['dataroot_gt']

        # 读取图像路径
        with open(self.opt['meta_info']) as fin:
            paths = [line.strip().split(' ')[0] for line in fin]
            self.paths = [osp.join(self.gt_folder, v) for v in paths]

        # 第二阶段退化参数
        self.kernel_range = [2 * v + 1 for v in range(3, 11)] # 7-21
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # 最终sinc核参数
        self.final_sinc_prob = opt['final_sinc_prob']
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        img_gt = imfrombytes(img_bytes, float32=True)
        #img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        img_gt =  cv2.resize(img_gt, (4720, 3146), interpolation=cv2.INTER_AREA)
        # 裁剪或填充为400x400
        # h, w = img_gt.shape[0:2]
        # crop_pad_size = 1600
        # if h < crop_pad_size or w < crop_pad_size:
        #     pad_h = max(0, crop_pad_size - h)
        #     pad_w = max(0, crop_pad_size - w)
        #     img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        #     top = random.randint(0, img_gt.shape[0] - crop_pad_size)
        #     left = random.randint(0, img_gt.shape[1] - crop_pad_size)
        #     img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------- 读取自定义 .mat 模糊核文件作为 kernel1 -------
        # mat_path = os.path.join('/data2/dataset/20250519/20250519HDR/', gt_path.split('/')[-1]).replace('.png', '_kernel_x2.mat')
        # mat_path = os.path.join('/home/sunqq/sqq/DKP/DIPDKP/data/log_DIPDKP/Set5_DIPDKP_lr_x4_DIPDKP', gt_path.split('/')[-1]).replace('.png', '.mat')
        mat_path = os.path.join('/data2/sqq/sunqq/sqq/KernelGAN/results', gt_path.split('/')[-1]).replace('.jpg', '_kernel_x2.mat')
        # mat_path = gt_path.replace('.jpg', '_kernel_x2.mat')
        if not osp.exists(mat_path):
            raise FileNotFoundError(f"Kernel .mat not found: {mat_path}")
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel1 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            mat_data = sio.loadmat(mat_path)
            kernel_np = mat_data.get('Kernel', None)
            if kernel_np is None:
                raise ValueError(f"Invalid kernel in {mat_path}")
            pad_size = (21 - 17) // 2
            kernel1 = np.pad(kernel_np, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel1 = torch.FloatTensor(kernel1)

        # ------- 随机生成 kernel2 -------
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2, self.kernel_prob2, kernel_size,
                self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2, self.betap_range2, noise_range=None
            )
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel_np, ((pad_size, pad_size), (pad_size, pad_size))) #kernel_np
        kernel2 = torch.FloatTensor(kernel2)
        # kernel2 = np.pad(kernel_np, ((pad_size, pad_size), (pad_size, pad_size)))
        # kernel2 = torch.FloatTensor(kernel2)
        # ------- 最终 sinc kernel -------
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        return {
            'gt': img_gt,
            'kernel1': kernel1,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


# @DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = {'type': 'disk'}
        self.source_root = opt['dataroot_source']
        self.target_root = opt['dataroot_target']

        # 读取路径列表
        with open(opt['meta_info']) as f:
            self.filenames = [line.strip().split('/')[-1] for line in f]
            self.source_paths = [osp.join(self.source_root, fn) for fn in self.filenames]
            self.target_paths = [osp.join(self.target_root, fn) for fn in self.filenames]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        source_path = self.source_paths[index]
        target_path = self.target_paths[index]

        retry = 3
        while retry > 0:
            try:
                source_bytes = self.file_client.get(source_path, 'source')
                target_bytes = self.file_client.get(target_path, 'target')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File error: {e}, retrying... ({retry - 1})')
                index = random.randint(0, len(self))
                source_path = self.source_paths[index]
                target_path = self.target_paths[index]
                retry -= 1
            else:
                break

        img_source = imfrombytes(source_bytes, float32=True)
        img_target = imfrombytes(target_bytes, float32=True)

        # optional resize or crop
        if self.opt.get('resize', False):
            size = self.opt['resize']
            img_source = cv2.resize(img_source, (size[0], size[1]), interpolation=cv2.INTER_AREA)
            img_target = cv2.resize(img_target, (size[0], size[1]), interpolation=cv2.INTER_AREA)

        # 转为 tensor 格式
        img_source = img2tensor([img_source], bgr2rgb=True, float32=True)[0]
        img_target = img2tensor([img_target], bgr2rgb=True, float32=True)[0]

        return {
            'img_source': img_source,
            'img_target': img_target,
            'source_path': source_path,
            'target_path': target_path
        }

    def __len__(self):
        return len(self.filenames)


# from torch.utils.data import DataLoader

# opt = {
#     'dataroot_gt': '/data2/sqq/blur_kernel/',  # e.g., '/data/train'
#     'meta_info': '/data2/dataset/20250519/20250519HDR/meta_info_20250519.txt',
#     'io_backend': {'type': 'disk'},
#     'use_hflip': True,
#     'use_rot': True,

#     # 第二次退化参数
#     'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
#     'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
#     'blur_sigma2': [0.2, 1.5],
#     'betag_range2': [0.5, 4],
#     'betap_range2': [1, 2],
#     'sinc_prob2': 0.1,
#     'final_sinc_prob': 0.8
# }

# dataset = TeleHyperDataset(opt)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# # Example usage:
# for batch in dataloader:
#     print(batch['gt'].shape)         # [B, C, H, W]
#     print(batch['kernel1'].shape)    # [B, 21, 21]
#     print(batch['kernel2'].shape)    # [B, 21, 21]
#     print(batch['sinc_kernel'].shape)
#     break

# import torchvision.utils as vutils
# import os

# save_dir = '/home/sunqq/sqq/Real-ESRGAN/output/processed_hdr/'
# os.makedirs(save_dir, exist_ok=True)

# for i, batch in enumerate(dataloader):
#     gts = batch['gt']  # Tensor: [B, C, H, W]
#     paths = batch['gt_path']  # List of strings

#     for j in range(gts.size(0)):
#         img_tensor = gts[j]  # [C, H, W]
#         img_name = os.path.basename(paths[j]).replace('.png', '_processed.png')

#         # 保存图像（注意还原RGB顺序并归一化到[0, 255]）
#         img = img_tensor.clone().detach()
#         img = img_tensor.detach().cpu().clamp(0, 1)  # 保证归一化
#         if img.dim() == 3:  # [C, H, W]
#             img = img.unsqueeze(0)  # -> [1, C, H, W]
#         img = img.cpu()
#         vutils.save_image(img, os.path.join(save_dir, img_name))

#     print(f"Saved batch {i + 1}")