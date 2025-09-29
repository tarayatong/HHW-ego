from basicsr.utils.options import parse_options
from torch.utils.data import DataLoader
from realesrgan.data.telehyper_dataset import TeleHyperDataset
from realesrgan.models.realesrnet_model import RealESRNetModel
import os
import cv2
import numpy as np
import torch
import pandas as pd
import random
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F
from realesrgan.data.telehyper_dataset import PairedImageDataset
from scripts.color_align import calculate_image_differences_rgb
from tqdm import tqdm
# 初始化数据集
opt_dataset = {
    'dataroot_gt': '/data2/sqq/blur_kernel/',  # e.g., '/data/train'
    'meta_info': '/data2/dataset/HSI_dataset/hdr_img_paths.txt', # _colorFreq '/home/sunqq/sqq/Real-ESRGAN/results/HDR_color/meta_info_color_20250519.txt', # ,'/data2/dataset/20250519/20250519HDR/meta_info_20250519.txt', #
    'io_backend': {'type': 'disk'},
    'use_hflip': True,
    'use_rot': True,
    'save_degraded': True,  # 启用保存功能
    'degraded_dir': 'results/degraded_pairs',  # 保存路径
     # 第二次退化参数
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.01, 0.25, 0.12, 0.03, 0.12, 0.03],
    'blur_sigma2': [0.0, 0.01],#[0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    'sinc_prob2': 0.1,   # 0.1
    'final_sinc_prob': 0.8,
}
import yaml

# 1. 直接加载配置
config_path = './options/train_realesrnet_x4plus.yml'
with open(config_path, 'r') as f:
    opt = yaml.safe_load(f)

# 2. 设置必要默认值
opt.setdefault('is_train', True)
opt.setdefault('dist', False)
opt['name'] = opt.get('name', 'telehyper_train')
opt['save_degraded'] = True  # 启用保存退化图像
opt['degraded_dir'] = '/data2/dataset/degtade_results/all/source'  # 保存路径
opt['training'] = False
opt['type_prob'] = [0.0, 0.0, 1.0]
opt['color_align'] = True
# 3. 初始化模型

def estimate_gaussian_noise(img):
    """
    估计高斯噪声强度（标准差）
    """
    # 转换为灰度 & 浮点型
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_gray = img.astype(np.float32)

    # 用高斯平滑后减去原图得到残差
    smoothed = cv2.GaussianBlur(img_gray, (3, 3), 0)
    residual = img_gray - smoothed

    # 计算残差的标准差
    sigma = np.std(residual)
    return sigma


def estimate_poisson_noise(img):
    """
    估计泊松噪声强度（均值-方差拟合系数）
    """
    # 转换为灰度 & 浮点型
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_gray = img.astype(np.float32)

    # 将图像划分为多个小块，统计每块的均值和方差
    block_size = 16
    h, w = img_gray.shape
    means = []
    variances = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img_gray[y:y+block_size, x:x+block_size]
            mean = np.mean(block)
            var = np.var(block)
            means.append(mean)
            variances.append(var)

    means = np.array(means)
    variances = np.array(variances)

    # 拟合：Var ≈ λ * Mean
    if np.sum(means) == 0:
        lambda_poisson = 0
    else:
        lambda_poisson = np.sum(variances) / np.sum(means)
    return lambda_poisson


def estimate_salt_pepper_noise(img):
    """
    估计椒盐噪声强度（极值像素比例）
    """
    # 转换为灰度
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    total_pixels = img_gray.size
    salt = np.sum(img_gray == 255)
    pepper = np.sum(img_gray == 0)
    noise_ratio = (salt + pepper) / total_pixels
    return noise_ratio


def estimate_noise_all(image_path):
    """
    同时估计高斯、泊松、椒盐噪声强度
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # print("❌ 无法读取图像：", image_path)
        return 5, 0.1, 1

    # print(f"📷 图像: {image_path}")

    gaussian_sigma = estimate_gaussian_noise(img)
    # print(f"🌿 高斯噪声估计 σ: {gaussian_sigma:.4f}")

    # poisson_lambda = estimate_poisson_noise(img)
    # # print(f"🌱 泊松噪声估计 λ: {poisson_lambda:.4f}")

    # sp_ratio = estimate_salt_pepper_noise(img)
    # print(f"🧂 椒盐噪声估计 比例: {sp_ratio*100:.2f}%")

    return gaussian_sigma#, poisson_lambda, sp_ratio

class TelehyperData():
    def __init__(self, opt):
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        self.save_degraded = opt.get('save_degraded', False)  # 新增配置项
        self.degraded_dir = opt.get('degraded_dir', '/home/sunqq/sqq/Real-ESRGAN/results/degraded')  # 保存路径

        if self.save_degraded and not os.path.exists(self.degraded_dir):
            os.makedirs(self.degraded_dir, exist_ok=True)
        self.device = 'cuda:0'
        self.opt = opt

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        self.gt = data['gt'].to(self.device)
        # USM sharpen the GT images
        if self.opt['gt_usm'] is True:
            self.gt = self.usm_sharpener(self.gt)

        self.kernel1 = data['kernel1'].to(self.device)
        self.kernel2 = data['kernel2'].to(self.device)
        self.sinc_kernel = data['sinc_kernel'].to(self.device)
        color_diff_dict = opt['color_diff_dict']

        ori_h, ori_w = self.gt.size()[2:4]
        # ----------------------- The first degradation process ----------------------- #
        updown_type = random.choices(['kernel', 'gaussian', 'poisson'], self.opt['type_prob'])[0]
        if updown_type == 'kernel':
            # blur
            out = filter2D(self.gt, self.kernel1)
            out = self.gt
        elif updown_type == 'gaussian':
            out = random_add_gaussian_noise_pt(
                self.gt, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=0.5)
        else:
            out = random_add_poisson_noise_pt(
                self.gt,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=0.5,
                clip=True,
                rounds=False)
        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        # gt_size = self.opt['gt_size']
        # self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
        if self.save_degraded:
            self._save_degraded_pairs(data['gt_path'], self.gt, self.lq)
        # training pair pool
        # self._dequeue_and_enqueue()
        # self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

    def _save_degraded_pairs(self, gt_paths, gt_tensor, lq_tensor):
        """保存HQ-LQ图像对"""
        for i in range(len(gt_paths)):
            # 获取文件名
            img_name = os.path.splitext(os.path.basename(gt_paths[i]))[0]

            # 转换tensor为numpy图像
            gt_img = tensor2img(gt_tensor[i:i+1], rgb2bgr=True, out_type=np.uint8)
            lq_img = tensor2img(lq_tensor[i:i+1], rgb2bgr=True, out_type=np.uint8)

            # 保存图像
            # cv2.imwrite(
            #     os.path.join(self.degraded_dir, f'{img_name}_HQ.png'),
            #     gt_img
            # )
            cv2.imwrite(
                os.path.join(self.degraded_dir, f'{img_name}_LQ.png'),
                lq_img
            )
            print(os.path.join(self.degraded_dir, f'{img_name}_LQ.png'))


def extract_parameter_ranges_quantile(
    df: pd.DataFrame,
    lower: float = 0.20,
    upper: float = 0.80,
    std_threshold: float = 3.0,
    alt_lower: float = 0.40,
    alt_upper: float = 0.60
) -> tuple:
    """
    提取颜色差异和噪声估计的参数分布范围（使用分位数，避免离群点影响）。
    若某列标准差超过阈值，使用40%-60%分位数避免离群值干扰。

    Returns:
        color_ranges_dict: dict[str, list]  # 每个颜色参数的[low, high]
        gaussian_sigma_range: list[float]   # 噪声参数的[low, high]
    """
    color_keys = [
        'brightness_diff_L_mean',
        'contrast_diff_L_std',
        'color_diff_a_mean',
        'color_diff_b_mean',
        'saturation_diff_S_mean'
    ]

    color_ranges_dict = {}
    for key in color_keys:
        std_val = df[key].std()
        if std_val > std_threshold:
            q_low, q_high = alt_lower, alt_upper
        else:
            q_low, q_high = lower, upper
        color_ranges_dict[key] = list(df[key].quantile([q_low, q_high]).values)

    # 对 gaussian_sigma 也使用类似判断（如有需要）
    sigma_std = df['gaussian_sigma'].std()
    if sigma_std > std_threshold:
        sigma_range = list(df['gaussian_sigma'].quantile([alt_lower, alt_upper]).values)
    else:
        sigma_range = list(df['gaussian_sigma'].quantile([lower, upper]).values)

    return color_ranges_dict, sigma_range

# 5. 训练循环
# for epoch in range(opt.get('epochs', 100)):
# model = TelehyperData(opt)

paired_dataset_opt = {
    'dataroot_source': '/data2/dataset/HSI_dataset/HDR_img',
    'dataroot_target': '/data2/dataset/HSI_dataset/Glasses_img',
    'meta_info': '/data2/dataset/HSI_dataset/glasses_img_paths.txt',
}

paired_dataset = PairedImageDataset(paired_dataset_opt)
dataloader = DataLoader(paired_dataset, batch_size=1, shuffle=False)
records = []
for i, data in enumerate(tqdm(dataloader)):
    src_path = data['source_path'][0]
    tgt_path = data['target_path'][0]
    img_src = data['img_source'][0].numpy().transpose(1, 2, 0)
    img_tgt = data['img_target'][0].numpy().transpose(1, 2, 0)

    # 颜色差异
    color_diff = calculate_image_differences_rgb(img_src, img_tgt)

    # 噪声估计
    sigma = estimate_noise_all(src_path)  # or pass img_src if needed

    record = {
        'source_path': src_path,
        'target_path': tgt_path,
        'gaussian_sigma': sigma,
        **color_diff
    }

    records.append(record)
    # if i>10:
    #     break

# 保存为 DataFrame
df = pd.DataFrame(records)
summary = df.describe(percentiles=[0.20, 0.80]).T
summary = summary[['mean', 'std', '20%', '80%']]
print(summary)
color_ranges, noise_range = extract_parameter_ranges_quantile(df, 0.40, 0.60, 2.5, 0.45, 0.55)
opt['color_diff_dict'] = color_ranges
opt['noise_range'] = noise_range

model = RealESRNetModel(opt)
dataset = TeleHyperDataset(opt_dataset)
dataloader = DataLoader(dataset, batch_size=opt.get('batch_size', 1))
num = 0
for data in dataloader:
    # lq_path = os.path.join('/data2/dataset/HSI_dataset/Glasses_img', os.path.basename(data['gt_path'][0])).replace('.png', '.jpg')
    # gaussian_sigma, poisson_lambda, sp_ratio = estimate_noise_all(lq_path)
    # opt['noise_range'] = [gaussian_sigma-2, gaussian_sigma+3]
    # opt['poisson_scale_range'] = [poisson_lambda-0.02, poisson_lambda+0.03]

    print(num)  # 打印数据的键
    model.feed_data(data)
    num += 1
    # if num>1:
    #     break