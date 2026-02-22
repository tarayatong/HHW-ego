"""
对齐HSI数据与实拍RGB图像的脚本。

支持两种方向：
1) photo_to_hsi：将实拍RGB图像对齐到HSI数据的RGB视图；
2) hsi_to_photo：将HSI立方体对齐到实拍RGB图像（按通道自动选择最优对齐），
   并输出矫正后的HSI数据及指定通道的可视化。
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
import os
from image_preprocessing import *

# ---------------------------------------------------------------------------
# 数据加载与工具函数
# ---------------------------------------------------------------------------

def load_hsi_package(pkl_path: Path) -> Dict:
    """读取HSI打包文件（dict）。"""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{pkl_path} 中的数据格式不是dict")

    return data


def extract_hsi_rgb(data: Dict) -> np.ndarray:
    """从数据包中提取RGB图像。"""
    if "rgb" not in data:
        raise KeyError("HSI数据包中缺少 `rgb` 键")

    rgb = data["rgb"]
    if not isinstance(rgb, np.ndarray):
        raise TypeError("数据字典中的 `rgb` 不是numpy数组")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"`rgb` 形状异常: {rgb.shape}")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


def extract_hsi_cube(data: Dict) -> np.ndarray:
    """从数据包中提取HSI立方体。"""
    if "hsi" not in data:
        raise KeyError("HSI数据包中缺少 `hsi` 键")

    cube = data["hsi"]
    if not isinstance(cube, np.ndarray):
        raise TypeError("数据字典中的 `hsi` 不是numpy数组")

    if cube.ndim != 3:
        raise ValueError(f"`hsi` 形状异常: {cube.shape}")

    if cube.dtype != np.float32:
        cube = cube.astype(np.float32)

    return cube


def extract_demosaic_gray(data: Dict) -> Optional[np.ndarray]:
    """提取去马赛克灰度参考图，若不存在则返回None。"""
    gray = data.get("demosaic_gray")
    if gray is None:
        return None

    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"`demosaic_gray` 形状异常: {gray.shape}")
    return gray


def remove_periodic_pattern(
    image: Union[np.ndarray, NDArray[np.float32]],
    period: int = 6,
    sample_origin: Optional[Tuple[int, int]] = (1066, 1575),
    sample_size: int = 18,
    blur_kernel_size: int = 11,
    use_random_origin: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    利用指定区域估计并移除图像中的周期性 6x6 纹理。

    返回 (校正后的图像, 周期kernel, 采样起点)。
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    period = int(period)
    if period <= 0:
        raise ValueError("period 必须为正整数")

    sample_size = int(sample_size)
    if sample_size < period or sample_size % period != 0:
        raise ValueError("sample_size 必须为 period 的倍数且不小于 period")

    if rng is None:
        rng = np.random.default_rng()

    if use_random_origin or sample_origin is None:
        max_y = h - sample_size
        max_x = w - sample_size
        if max_y < 0 or max_x < 0:
            raise ValueError("sample_size 超出图像范围")
        sample_y = rng.integers(0, max_y + 1)
        sample_x = rng.integers(0, max_x + 1)
    else:
        sample_y, sample_x = sample_origin

    if not (0 <= sample_y <= h - sample_size and 0 <= sample_x <= w - sample_size):
        raise ValueError("sample_origin 超出图像范围")

    patch = img[
        sample_y : sample_y + sample_size,
        sample_x : sample_x + sample_size,
    ]

    smooth_ref = cv2.GaussianBlur(
        patch,
        (blur_kernel_size, blur_kernel_size),
        sigmaX=0,
        sigmaY=0,
    )

    epsilon = 1e-6
    pattern = patch / (smooth_ref + epsilon)

    kernel = np.zeros((period, period), dtype=np.float32)
    counts = np.zeros((period, period), dtype=np.int32)

    for y in range(sample_size):
        for x in range(sample_size):
            kernel[y % period, x % period] += pattern[y, x]
            counts[y % period, x % period] += 1

    counts = np.where(counts == 0, 1, counts)
    kernel /= counts

    kernel_mean = float(np.mean(kernel))
    if kernel_mean > 0:
        kernel /= kernel_mean

    yy, xx = np.indices((h, w))
    pattern_map = kernel[(yy - sample_y) % period, (xx - sample_x) % period]

    corrected = img / np.maximum(pattern_map, epsilon)

    corrected = np.clip(corrected, 0, None)

    return corrected.astype(np.float32), kernel, (int(sample_y), int(sample_x))


def ensure_bgr(image: np.ndarray, assume_rgb: bool = False) -> np.ndarray:
    """确保图像为BGR三通道格式。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"无法转换为BGR格式，输入形状: {image.shape}")

    if assume_rgb:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def imread_unicode(path: Path, flag: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """支持Unicode路径的图像读取。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法打开图像文件: {path}") from exc

    image = cv2.imdecode(buffer, flag)
    return image


def build_feature_detector(
    method: str, max_features: int
) -> Tuple[cv2.Feature2D, int]:
    """
    创建特征点检测器以及对应的描述符距离度量。

    返回 (detector, norm_type)。
    """
    method = method.lower()

    if method == "sift" or method == "auto":
        if hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create(nfeatures=max_features)
            return detector, cv2.NORM_L2
        if method == "sift":
            raise RuntimeError("当前OpenCV未编译SIFT，请尝试ORB或安装opencv-contrib-python")

    if method == "orb" or method == "auto":
        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=5)
        return detector, cv2.NORM_HAMMING

    raise ValueError(f"不支持的特征方法: {method}")


def match_keypoints(
    desc1: np.ndarray,
    desc2: np.ndarray,
    norm_type: int,
    ratio_test: float,
) -> list:
    """使用比值测试匹配特征点。"""
    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    return good_matches


def estimate_homography(
    kp_src,
    kp_dst,
    matches,
    ransac_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据匹配估计单应性矩阵。"""
    if len(matches) < 4:
        raise RuntimeError("有效匹配不足，无法估计单应性矩阵")

    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    # if H is None:
    #     raise RuntimeError("findHomography 失败，检查输入匹配质量")

    return H, mask.ravel().astype(bool)


def stretch_to_uint8(
    image: np.ndarray, clip_percentiles: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    将浮点图像拉伸到uint8，抑制极端值。

    clip_percentiles 为 [low, high] 百分位，默认 1%-99%。
    """
    low, high = np.percentile(image, clip_percentiles)
    if np.isclose(high, low):
        return np.zeros_like(image, dtype=np.uint8)

    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """将任意二维/三通道图像转换为8位灰度图。"""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3:
        if array.shape[2] == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 1:
            gray = array[:, :, 0]
        else:
            raise ValueError(f"无法处理通道数为 {array.shape[2]} 的图像")
    else:
        raise ValueError(f"无法处理维度为 {array.ndim} 的图像")

    if gray.dtype == np.uint8:
        return gray

    gray = gray.astype(np.float32, copy=False)
    return stretch_to_uint8(gray)

def gray_pic_pre_process(tgt_gray, src_gray):

    src_gray = src_gray[::6,::6]
    # src_gray = cv2.resize(src_gray, (tgt_gray.shape[1], tgt_gray.shape[0]))
    # src_gray = src_gray[::3,::3]
    # src_gray = cv2.resize(src_gray, (tgt_gray.shape[1], tgt_gray.shape[0]))
    src_gray = basic_denoise_and_edge_enhance(src_gray)
    # src_gray = match_histograms(src_gray, tgt_gray, channel_axis=-1)
    src_gray = cv2.GaussianBlur(src_gray, (3, 3), 0)
    src_gray = cv2.filter2D(src_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    src_gray = basic_denoise_and_edge_enhance(src_gray)

    # src_gray = cv2.GaussianBlur(src_gray[::3, ::3], (3, 3), 0)
    # src_gray = cv2.filter2D(src_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    tgt_gray = tgt_gray[::7, ::7]
    tgt_gray = cv2.resize(tgt_gray, (src_gray.shape[1], src_gray.shape[0]))
    # tgt_gray = cv2.GaussianBlur(tgt_gray, (5, 5), 0)
    # tgt_gray = cv2.filter2D(tgt_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    # tgt_gray = cv2.filter2D(tgt_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    return tgt_gray, src_gray

def align_image_to_target(
    source_image: Dict,
    target_image: np.ndarray,
    method: str = "auto",
    max_features: int = 3000,
    ratio_test: float = 0.75,
    ransac_thresh: float = 5.0,
    warp_flags: int = cv2.INTER_LINEAR,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    利用特征点匹配将 source_image 对齐到 target_image。

    返回对齐后的图像，若 return_homography=True 则返回 (对齐图, H)。
    """
    aligned_hsi = {}
    source_gray = source_image['demosaic_gray']
    detector, norm_type = build_feature_detector(method, max_features)

    
    tgt_gray = to_gray_uint8(target_image)
    src_gray = to_gray_uint8(source_gray)
    tgt_gray, src_gray = gray_pic_pre_process(tgt_gray, src_gray)

    kp_src, des_src = detector.detectAndCompute(src_gray, None)
    kp_tgt, des_tgt = detector.detectAndCompute(tgt_gray, None)

    if des_src is None or des_tgt is None:
        aligned_hsi['demosaic_gray'] = np.zeros_like(source_gray, dtype=np.uint8)
        return aligned_hsi
    bfmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bfmatcher.knnMatch(des_src, des_tgt, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:   # Lowe ratio test
            good.append(m)
    matches = good
    # matches = match_keypoints(des_src, des_tgt, norm_type, ratio_test)
    if len(matches) < 15:
        aligned_hsi['demosaic_gray'] = np.zeros_like(source_gray, dtype=np.uint8)
        return aligned_hsi
    else:
        matches = sorted(matches, key=lambda x: x.distance)[:15]

    H, _ = estimate_homography(kp_src, kp_tgt, matches, ransac_thresh)
    if H is None:
        aligned_hsi['demosaic_gray'] = np.zeros_like(source_gray, dtype=np.uint8)
        return aligned_hsi

    target_h, target_w = tgt_gray.shape[:2]
    for keys, values in source_image.items():
        try:
            aligned_hsi[keys] = cv2.warpPerspective(
                values,
                H,
                (target_w, target_h),
                flags=warp_flags,
            )
        except Exception as e:
            # print(f"Error aligning {keys}: {e}")
            aligned_hsi[keys] = source_image[keys]
    return aligned_hsi

hsi_file_list = os.listdir("HSI_dataset/HSI_data")
# rgb_file_list = os.listdir("exp_results/RealESRGAN_enhanced")
# assert len(hsi_file_list) == len(rgb_file_list), "HSI和RGB文件数量不一致"
for hsi_file in hsi_file_list:
    name_id = hsi_file.split(".")[0]
    if os.path.exists(f"HSI_aligned/{name_id}.pkl"):
        # print(f"HSI_aligned/{name_id}.pkl already exists")
        continue
    # if int(name_id) < 37:
    #     continue
    print(f"Processing {name_id}")
    hsi_data = pickle.load(open(f"HSI_dataset/HSI_data/{hsi_file}", "rb"))
    rgb_data = cv2.imread(f"exp_results/RealESRGAN_enhanced/{name_id}_out.jpg")
    aligned_hsi= align_image_to_target(hsi_data, rgb_data, method="sift", max_features=3000, ratio_test=0.75, ransac_thresh=5.0, warp_flags=cv2.INTER_LINEAR)
    if aligned_hsi['demosaic_gray'].sum()==0:
        continue
    cv2.imwrite(f"HSI_aligned/{name_id}_gray.png", aligned_hsi['demosaic_gray'])
    pickle.dump(aligned_hsi, open(f"HSI_aligned/{name_id}.pkl", "wb"))
# hsi_data = pickle.load(open("HSI_dataset/HSI_data/169.pkl", "rb"))
# hsi_gray = hsi_data['demosaic_gray']
# hsi_cube = hsi_data['hsi']
# rgb_data = cv2.imread("exp_results/RealESRGAN_enhanced/169_out.jpg")

# aligned_gray, aligned_hsi, H = align_image_to_target(hsi_cube, rgb_data, hsi_gray, method="sift", max_features=3000, ratio_test=0.75, ransac_thresh=5.0, warp_flags=cv2.INTER_LINEAR, return_homography=False)

# cv2.imwrite("HSI_aligned/aligned_gray.png", aligned_gray)
# pickle.dump(aligned_hsi, open("HSI_aligned/aligned_hsi.pkl", "wb"))
