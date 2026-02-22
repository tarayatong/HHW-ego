"""
HSI色彩校正与色域增强工具 (Enhanced Gamut Version)

主要改进：
-----------
1. 扩展色域支持：
   - 允许超出标准sRGB色域范围
   - 使用xyY色彩空间分离处理色度和亮度
   - 保留HSI提供的丰富色彩信息

2. 色度增强：
   - 在xyY空间中增强色彩饱和度
   - 向HSI色度方向引导，获得更真实的色彩
   - 可调节的增强强度（chroma_boost参数）

3. 改进的色调映射：
   - 使用Reinhard tone mapping替代简单归一化
   - 保留色度信息，只压缩亮度
   - 避免色彩失真和过度压缩

4. 自适应处理：
   - 移除硬性clip操作
   - 使用软压缩函数处理超出范围的值
   - 在最终输出时才进行必要的范围限制

使用方法：
---------
调整以下三个关键参数以控制色域增强效果：

- chroma_boost (1.0-2.0): 色彩饱和度增强
  * 1.0 = 不增强
  * 1.5 = 推荐（适度增强）
  * 2.0 = 最大增强

- tone_mapping_max (0.95-1.5): 亮度动态范围
  * 0.95 = 标准范围
  * 1.2 = 推荐（允许HDR效果）
  * 1.5 = 最大动态范围

- enable_gamut_extension (True/False): 是否启用扩展色域
  * True = 允许超出sRGB（推荐）
  * False = 限制在sRGB内

色域分析：
---------
使用color_analysis.py工具分析增强前后的色域变化：
- xy色度图：查看色彩覆盖范围
- Lab ab分布：查看色彩饱和度
- 色域体积：量化色域大小
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 不显示图形界面，直接保存
import matplotlib.pyplot as plt
import pickle
from colour.models import RGB_COLOURSPACE_sRGB
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER # Corrected import
from scipy.interpolate import interp1d
import cv2
from PIL import Image
from skimage import exposure, img_as_float32
import os

# --- 🌈 色彩校正（RGB <-> XYZ） ---
def rgb_to_xyz(rgb_img):
    rgb_to_xyz_matrix = RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ
    def linearize_srgb(rgb):
        mask = rgb <= 0.04045
        linear_rgb = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return linear_rgb
    linear_rgb = linearize_srgb(rgb_img)
    xyz_img = np.tensordot(linear_rgb, rgb_to_xyz_matrix, axes=([2], [1]))
    return xyz_img

def xyz_to_rgb(xyz_img, clip_output=True, extended_gamut=False):
    """
    XYZ转RGB，支持扩展色域模式
    
    参数:
    - clip_output: 是否将输出限制在[0,1]，False时允许超出sRGB色域
    - extended_gamut: 是否使用扩展色域处理（保留更多色彩信息）
    """
    xyz_to_rgb_matrix = RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB
    _ = clip_output  # 参数保留兼容，但不再执行裁剪
    _ = extended_gamut
    linear_rgb = np.tensordot(xyz_img, xyz_to_rgb_matrix, axes=([2], [1]))
    
    def gamma_encode(linear_rgb):
        linear_rgb = np.nan_to_num(linear_rgb, nan=0.0)
        mask = linear_rgb <= 0.0031308
        srgb = np.where(mask, 12.92 * linear_rgb, 1.055 * (linear_rgb ** (1/2.4)) - 0.055)
        return srgb
    
    rgb_img = gamma_encode(linear_rgb)
    return rgb_img

def adjust_rgb_by_cct(rgb_img, target_cct_map, original_cct_map=None, strength=1.0, blur_kernel_size=(25, 25)):
    if original_cct_map is None:
        original_xyz = rgb_to_xyz(rgb_img)
        original_cct_map = xyz_to_cct(original_xyz)
    xyz_img = rgb_to_xyz(rgb_img)
    ratio = np.ones_like(target_cct_map)
    valid_mask = (target_cct_map > 0) & (original_cct_map > 0) & \
                 np.isfinite(target_cct_map) & np.isfinite(original_cct_map)

    ratio[valid_mask] = (original_cct_map[valid_mask] / target_cct_map[valid_mask]) ** strength

    ratio = np.nan_to_num(ratio, nan=1.0)
        # 确保 blur_kernel_size 是奇数
    k_h = blur_kernel_size[0] if blur_kernel_size[0] % 2 != 0 else blur_kernel_size[0] + 1
    k_w = blur_kernel_size[1] if blur_kernel_size[1] % 2 != 0 else blur_kernel_size[1] + 1

    # 对 ratio 进行高斯平滑
    ratio_smoothed = cv2.GaussianBlur(ratio, (k_w, k_h), 0) # sigmaX和sigmaY为0，表示根据核大小自动计算

    ratio = ratio_smoothed

    adjusted_xyz = xyz_img * ratio[:, :, np.newaxis]
    adjusted_rgb = xyz_to_rgb(adjusted_xyz)
    return adjusted_rgb

def xyz_to_cct(xyz_img):
    xyz = np.clip(xyz_img, 0, None)
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    denom = X + Y + Z
    valid = (denom > 1e-6)

    # 先滤掉不合理的 xy
    x = np.zeros_like(X)
    y = np.zeros_like(Y)
    x[valid] = X[valid] / denom[valid]
    y[valid] = Y[valid] / denom[valid]

    valid &= (x > 0) & (y > 0) & (x + y < 1)
    valid &= np.abs(0.1858 - y) > 1e-3  # 离奇异点远一点
    valid &= (Y > 1e-3)                 # 排除几乎无亮度的点

    cct = np.full_like(X, np.nan, dtype=np.float32)
    cct[valid] = xy_to_cct(np.stack([x[valid], y[valid]], axis=-1))

    # 对结果再做合理区间裁剪
    cct = np.clip(cct, 1000, 20000)

    # cct = xy_to_cct(np.stack([X / denom, Y / denom], axis=-1))

    return cct


def xy_to_cct(xy_img, safety_margin=1e-3):
    """
    根据xy色度坐标估算相关色温（CCT），对奇异点或无效点返回np.nan。

    参数:
        xy_img: (..., 2) 或 (..., 3) 数组，最后两个通道为x、y。
        safety_margin: y值与奇异点0.1858之间的最小安全距离。

    返回:
        CCT数组，其中无效点被标记为np.nan。
    """
    if xy_img.ndim < 2:
        raise ValueError(f"xy_img维度不足，shape={xy_img.shape}")
    if xy_img.shape[-1] < 2:
        raise ValueError(f"xy_img最后一维需至少包含x、y两个通道，收到shape={xy_img.shape}")

    x = xy_img[..., 0]
    y = xy_img[..., 1]

    # 初始化CCT数组，默认全部为无效值
    cct = np.full(x.shape, np.nan, dtype=np.float32)

    # 构建一个严格的有效性掩码
    valid_mask = np.isfinite(x) & np.isfinite(y)
    valid_mask &= (np.abs(0.1858 - y) > safety_margin)

    # 仅对通过掩码的有效像素进行计算
    if np.any(valid_mask):
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        n = (x_valid - 0.3320) / (0.1858 - y_valid)
        
        cct_values = 437 * (n ** 3) + 3601 * (n ** 2) + 6861 * n + 5517
        
        # 将计算出的有效值放回数组的对应位置
        cct[valid_mask] = cct_values
        
    return cct

def xyz_to_xyY(xyz_img):
    """XYZ转换到xyY色彩空间，分离色度和亮度"""
    X = xyz_img[..., 0]
    Y = xyz_img[..., 1]
    Z = xyz_img[..., 2]
    
    sum_xyz = X + Y + Z + 1e-12
    x = X / sum_xyz
    y = Y / sum_xyz
    
    return np.stack([x, y, Y], axis=-1)

def xyY_to_xyz(xyY_img):
    """xyY转换回XYZ色彩空间"""
    x = xyY_img[..., 0]
    y = xyY_img[..., 1]
    Y = xyY_img[..., 2]
    
    # 避免除零
    y_safe = np.where(y > 1e-12, y, 1e-12)
    
    X = (Y / y_safe) * x
    Z = (Y / y_safe) * (1 - x - y)
    
    return np.stack([X, Y, Z], axis=-1)

def enhance_chroma_in_xyY(xyY_img, hsi_xyY_img, chroma_boost=1.3):
    """
    在xyY空间中以区域（整体）统计的方式增强色度，保持亮度不变
    
    参数:
    - xyY_img: RGB转换的xyY图像
    - hsi_xyY_img: HSI转换的xyY图像
    - chroma_boost: 色度增强系数 (>1.0增强色彩饱和度)
    """
    # D65白点（标准日光）
    D65_x, D65_y = 0.3127, 0.3290
    
    # 计算色度偏移（相对于白点）
    rgb_dx = xyY_img[..., 0] - D65_x
    rgb_dy = xyY_img[..., 1] - D65_y
    
    hsi_dx = hsi_xyY_img[..., 0] - D65_x
    hsi_dy = hsi_xyY_img[..., 1] - D65_y
    
    # HSI的色度距离（饱和度的一种度量）
    hsi_chroma_dist = np.sqrt(hsi_dx**2 + hsi_dy**2)
    rgb_chroma_dist = np.sqrt(rgb_dx**2 + rgb_dy**2)
    
    # 基于整幅图像的加权平均色度差异（使用HSI色度距离作为权重）
    weights = hsi_chroma_dist + 1e-8
    # 基于色相分布的平衡：对占比过大的色相降低权重，防止单一暖色主导
    hue_angles = np.mod(np.arctan2(hsi_dy, hsi_dx), 2 * np.pi)
    hue_bins = 36
    bin_edges = np.linspace(0.0, 2 * np.pi, hue_bins + 1)
    bin_indices = np.digitize(hue_angles, bin_edges, right=False) - 1
    bin_indices = np.where(bin_indices < 0, 0, bin_indices)
    bin_indices = np.where(bin_indices >= hue_bins, hue_bins - 1, bin_indices)
    bin_counts = np.bincount(bin_indices.ravel(), minlength=hue_bins).astype(np.float32)
    bin_counts[bin_counts == 0] = 1.0
    hue_balance = 1.0 / bin_counts[bin_indices]
    low_chroma_mask = hsi_chroma_dist < 1e-4
    hue_balance[low_chroma_mask] = 1.0
    hue_balance /= np.mean(hue_balance)
    weights *= hue_balance

    # 亮度加权，避免暗部对整体偏移影响过大，同时限制亮部权重上限
    luminance = xyY_img[..., 2]
    luminance_norm = luminance / (np.mean(luminance) + 1e-6)
    luminance_weights = np.where(luminance_norm < 0.3, 0.3, luminance_norm)
    luminance_weights = np.where(luminance_weights > 2.0, 2.0, luminance_weights)
    weights *= luminance_weights
    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        weight_sum = np.prod(xyY_img.shape[:2])
        weights = np.ones_like(weights)
    
    mean_rgb_dx = np.sum(rgb_dx * weights) / weight_sum
    mean_rgb_dy = np.sum(rgb_dy * weights) / weight_sum
    mean_hsi_dx = np.sum(hsi_dx * weights) / weight_sum
    mean_hsi_dy = np.sum(hsi_dy * weights) / weight_sum
    
    # 计算区域级别的色度偏移，并将其应用到所有像素
    shift_dx = (mean_hsi_dx - mean_rgb_dx) * 0.5
    shift_dy = (mean_hsi_dy - mean_rgb_dy) * 0.5

    # 根据整体色度差异动态缩放区域偏移，防止一次性偏移过大
    mean_rgb_chroma = np.mean(rgb_chroma_dist)
    mean_hsi_chroma = np.mean(hsi_chroma_dist)
    if mean_rgb_chroma > 0:
        raw_scale = mean_hsi_chroma / (mean_rgb_chroma + 1e-6)
        if raw_scale < 0.0:
            raw_scale = 0.0
        if raw_scale > 1.0:
            raw_scale = 1.0
        global_shift_scale = raw_scale
    else:
        global_shift_scale = 1.0
    shift_dx *= global_shift_scale
    shift_dy *= global_shift_scale

    target_dx = rgb_dx + shift_dx
    target_dy = rgb_dy + shift_dy

    # 限制每个像素的色度偏移幅度，避免落到色度图边界
    max_chroma_offset = max(np.percentile(hsi_chroma_dist, 95), 1e-3)
    target_magnitude = np.sqrt(target_dx**2 + target_dy**2)
    clip_scale = np.minimum(1.0, max_chroma_offset / (target_magnitude + 1e-8))
    target_dx *= clip_scale
    target_dy *= clip_scale

    enhanced_dx = target_dx * chroma_boost
    enhanced_dy = target_dy * chroma_boost

    enhanced_mag = np.sqrt(enhanced_dx**2 + enhanced_dy**2)
    clip_scale_boost = np.minimum(1.0, max_chroma_offset / (enhanced_mag + 1e-8))
    enhanced_dx *= clip_scale_boost
    enhanced_dy *= clip_scale_boost

    threshold = 0.1
    abs_dx = np.abs(enhanced_dx)
    abs_dy = np.abs(enhanced_dy)
    reduction_dx = np.where(abs_dx > threshold, threshold / (abs_dx + 1e-8), 1.0)
    reduction_dy = np.where(abs_dy > threshold, threshold / (abs_dy + 1e-8), 1.0)
    enhanced_dx = enhanced_dx * reduction_dx
    enhanced_dy = enhanced_dy * reduction_dy
    
    # 计算新的xy坐标
    min_chroma = 1e-3
    max_chroma = 1.0 - min_chroma
    raw_x = D65_x + enhanced_dx
    raw_y = D65_y + enhanced_dy
    enhanced_x = np.where(raw_x < min_chroma, min_chroma, raw_x)
    enhanced_x = np.where(enhanced_x > max_chroma, max_chroma, enhanced_x)
    enhanced_y = np.where(raw_y < min_chroma, min_chroma, raw_y)
    enhanced_y = np.where(enhanced_y > max_chroma, max_chroma, enhanced_y)

    sum_xy = enhanced_x + enhanced_y
    overflow_mask = sum_xy > max_chroma
    if np.any(overflow_mask):
        scale_xy = max_chroma / (sum_xy[overflow_mask] + 1e-8)
        enhanced_x[overflow_mask] *= scale_xy
        enhanced_y[overflow_mask] *= scale_xy
    
    # 保持原始亮度Y，只改变色度xy
    enhanced_xyY = np.stack([enhanced_x, enhanced_y, xyY_img[..., 2]], axis=-1)
    
    return enhanced_xyY

def adaptive_tone_mapping(xyz_img, target_max=0.95, preserve_color=True):
    """
    自适应色调映射，保留色彩信息同时压缩动态范围
    
    参数:
    - xyz_img: XYZ图像
    - target_max: 目标最大亮度（Y通道）
    - preserve_color: 是否保持色度不变
    """
    if preserve_color:
        # 转换到xyY空间，只调整Y
        xyY = xyz_to_xyY(xyz_img)
        Y = xyY[..., 2]
        
        # Reinhard tone mapping
        Y_max = np.percentile(Y, 99.5)  # 使用99.5分位数而不是最大值，避免极值影响
        if Y_max > target_max:
            # 应用Reinhard tonemapping
            L_white = Y_max * 1.5
            Y_mapped = Y * (1 + Y / (L_white ** 2)) / (1 + Y)
            Y_mapped = Y_mapped * (target_max / np.max(Y_mapped))
        else:
            Y_mapped = Y
        
        # 更新Y通道
        xyY[..., 2] = Y_mapped
        
        # 转换回XYZ
        return xyY_to_xyz(xyY)
    else:
        # 简单的全局缩放
        Y = xyz_img[..., 1]
        Y_max = np.max(Y)
        if Y_max > target_max:
            scale = target_max / Y_max
            return xyz_img * scale
        return xyz_img

def hsi_to_xyz(hsi, wavelengths):
    cmf = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    x_bar = interp1d(cmf.wavelengths, cmf.values[:, 0], bounds_error=False, fill_value=0)(wavelengths)
    y_bar = interp1d(cmf.wavelengths, cmf.values[:, 1], bounds_error=False, fill_value=0)(wavelengths)
    z_bar = interp1d(cmf.wavelengths, cmf.values[:, 2], bounds_error=False, fill_value=0)(wavelengths)

    delta_lambda = np.diff(wavelengths, append=wavelengths[-1] + (wavelengths[-1]-wavelengths[-2]))

    X = np.sum(hsi * x_bar[np.newaxis, np.newaxis, :] * delta_lambda[np.newaxis, np.newaxis, :], axis=2)
    Y = np.sum(hsi * y_bar[np.newaxis, np.newaxis, :] * delta_lambda[np.newaxis, np.newaxis, :], axis=2)
    Z = np.sum(hsi * z_bar[np.newaxis, np.newaxis, :] * delta_lambda[np.newaxis, np.newaxis, :], axis=2)

    xyz_img = np.stack([X, Y, Z], axis=-1)
    xyz_img = np.clip(xyz_img, 0, None)
    scale = np.percentile(xyz_img[..., 1], 99)
    if scale > 0:
        xyz_img /= scale
    return xyz_img

def spectral_entropy(band):
    histogram, _ = np.histogram(band.flatten(), bins=1024, range=(band.min(), band.max()+10), density=True)
    histogram += 1e-12
    entropy = -np.sum(histogram * np.log2(histogram))
    return entropy

# 修改 select_bands_pca_entropy，使其能接收一个有效波段列表，并考虑平均强度
def select_bands_pca_entropy(hsi_cube, n_bands=15, plot=False, valid_band_indices=None):
    H, W, C = hsi_cube.shape

    # 如果指定了有效波段，则只处理这些波段
    if valid_band_indices is not None:
        # 过滤掉不在有效波段列表中的波段
        temp_hsi_cube = np.zeros((H, W, len(valid_band_indices)), dtype=hsi_cube.dtype)
        for i, idx in enumerate(valid_band_indices):
            temp_hsi_cube[:, :, i] = hsi_cube[:, :, idx]
        hsi_cube_filtered = temp_hsi_cube

        original_indices_map = np.array(valid_band_indices) # 映射回原始HISCUBE的索引
        num_effective_bands = hsi_cube_filtered.shape[2]

        if num_effective_bands == 0:
            print("Warning: No valid bands available for selection in this patch.")
            return [] # 返回空列表，表示没有可选波段

        hsi_reshaped = hsi_cube_filtered.reshape(-1, num_effective_bands)
    else: # 如果没有提供 valid_band_indices，则处理所有波段 (通常不应该发生在此处调用)
        hsi_reshaped = hsi_cube.reshape(-1, C)
        num_effective_bands = C
        original_indices_map = np.array(range(C))
        hsi_cube_filtered = hsi_cube

    if hsi_reshaped.shape[0] == 0: # 检查图像块是否为空
        print("Warning: HSI patch is empty, cannot select bands.")
        return []

    # 检查 hsi_reshaped 的行数是否小于列数，这会导致 PCA 协方差计算问题
    if hsi_reshaped.shape[0] < num_effective_bands:
        print(f"Warning: Data size ({hsi_reshaped.shape[0]} pixels) is too small for PCA on {num_effective_bands} bands in this patch. Using entropy and intensity only for band selection.")

        entropy_scores = np.array([spectral_entropy(hsi_cube_filtered[:, :, i]) for i in range(num_effective_bands)])
        intensity_scores = np.array([np.mean(hsi_cube_filtered[:, :, i]) for i in range(num_effective_bands)])

        if intensity_scores.max() > 0:
            intensity_scores_norm = intensity_scores / intensity_scores.max()
        else:
            intensity_scores_norm = np.zeros_like(intensity_scores)

        # 确保 combined_score 不会因为熵或强度为0而导致全部为0
        combined_score = entropy_scores * (intensity_scores_norm + 1e-6)

        best_band_indices_relative = np.where(entropy_scores > 0)[0]
        if len(best_band_indices_relative) == 0:
            best_band_indices_relative = np.argsort(-combined_score)[:n_bands]
        elif len(best_band_indices_relative) > n_bands:
            best_band_indices_relative = best_band_indices_relative[np.argsort(-combined_score[best_band_indices_relative])[:n_bands]]

        return sorted(original_indices_map[best_band_indices_relative].tolist())

    # # Step1: PCA 主成分贡献
    # try:
    #     cov = np.cov(hsi_reshaped.T)
    #     eig_vals, eig_vecs = np.linalg.eigh(cov)
    #     pca_contribution = np.abs(eig_vecs).sum(axis=1)
    # except np.linalg.LinAlgError:
    #     print("Warning: Could not compute PCA contribution for this patch. Falling back to entropy and intensity only.")
    #     pca_contribution = np.ones(num_effective_bands)

    # Step2: 光谱熵
    entropy_scores = np.array([spectral_entropy(hsi_cube_filtered[:, :, i]) for i in range(num_effective_bands)])

    # Step3: 波段平均强度 (光响应强度)
    intensity_scores = np.array([np.mean(hsi_cube_filtered[:, :, i]) for i in range(num_effective_bands)])
    if intensity_scores.max() > 0:
        intensity_scores_norm = intensity_scores / intensity_scores.max()
    else:
        intensity_scores_norm = np.zeros_like(intensity_scores)

    # Step4: 联合评分 (PCA贡献 * 熵 * 强度)
    combined_score = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min()) * (intensity_scores_norm + 1e-6)

    if np.all(entropy_scores == 0) or np.all(intensity_scores_norm == 0):
        print("Warning: All bands in this patch have zero entropy or zero intensity. No effective bands to select.")
        return []

    best_band_indices_relative = np.where(entropy_scores > 0)[0]

    if len(best_band_indices_relative) == 0:
        best_band_indices_relative = np.argsort(-combined_score)[:n_bands]
    elif len(best_band_indices_relative) > n_bands:
        best_band_indices_relative = best_band_indices_relative[np.argsort(-combined_score[best_band_indices_relative])[:n_bands]]

    best_band_indices_relative = best_band_indices_relative[best_band_indices_relative < num_effective_bands]

    if plot:
        plt.figure(figsize=(10, 4))
        # plt.plot(pca_contribution, label="PCA Contribution (Effective Bands)")
        plt.plot(entropy_scores, label="Spectral Entropy (Effective Bands)")
        plt.plot(intensity_scores_norm, label="Normalized Intensity (Effective Bands)")
        plt.plot(combined_score, label="Combined Score (Effective Bands)", linewidth=2)
        plt.scatter(best_band_indices_relative, combined_score[best_band_indices_relative], color='red', label="Selected Bands (Relative)")
        plt.xlabel("Relative Band Index")
        plt.ylabel("Score")
        plt.title("Band Selection (PCA + Entropy + Intensity)")
        plt.legend()
        plt.show()

    selected_global_indices = original_indices_map[best_band_indices_relative].tolist()
    return sorted(selected_global_indices)

# 新增：只检测坏波段，不进行插值
def detect_bad_bands_only(hsi_data, wavelengths, min_std_threshold_factor=0.01, max_std_threshold_factor=4., min_corr_threshold=0.7, avg_intensity_diff_threshold_factor=0.5, min_avg_intensity_factor=0.12):
    num_bands = hsi_data.shape[2]
    bad_band_indices = set()

    all_stds = [np.std(hsi_data[:,:,i]) for i in range(num_bands)]
    # 过滤掉0值的标准差，避免平均值被拉低
    valid_stds = [s for s in all_stds if s > 0]
    if len(valid_stds) > 0:
        mean_std_all_bands = np.mean(valid_stds)
    else: # 如果所有波段标准差都为0，则无法设置动态阈值，设一个非常小的固定值
        mean_std_all_bands = 1e-6
        print("Warning: All bands have zero standard deviation. Using a default low threshold.")

    min_std_threshold = mean_std_all_bands * min_std_threshold_factor
    max_std_threshold = mean_std_all_bands * max_std_threshold_factor

    # print(f"Detecting bad bands. Min Std Dev Threshold: {min_std_threshold:.4f}")
    # 计算所有波段的平均强度
    all_avg_intensities = [np.mean(hsi_data[:,:,i]) for i in range(num_bands)]
    min_avg_intensity_threshold = np.mean(all_avg_intensities)*min_avg_intensity_factor
    for i in range(num_bands):
        band = hsi_data[:, :, i]
        band_flat = band.flatten()
        current_avg_intensity = all_avg_intensities[i]
        # --- 新增条件：排除平均强度过低的波段 ---
        if current_avg_intensity < min_avg_intensity_threshold:
            bad_band_indices.add(i)
            # print(f"1 Flagged band {i} ({wavelengths[i]:.1f}nm) (Avg Intensity: {current_avg_intensity:.4f} < {min_avg_intensity_threshold:.4f}) - Reason: Too low average intensity") # For debugging
            continue
        # 检查标准差：过低可能表示信息缺失或均匀噪声
        # if np.std(band_flat) > max_std_threshold:
        #     bad_band_indices.add(i)
        #     print(f"  Flagged band {i} ({wavelengths[i]:.1f}nm) (Std Dev: {np.std(band_flat):.4f} > {max_std_threshold:.4f}) - Reason: High Std Dev")
        #     continue
        # 检查是否所有像素都相同（或接近），避免全零波段
        if np.all(np.isclose(band_flat, band_flat[0], atol=1e-6)):
            bad_band_indices.add(i)
            # print(f"3 Flagged band {i} ({wavelengths[i]:.1f}nm) - Reason: Uniform (all pixels almost same)")
            continue

        # 检查相关性：与相邻波段相关性过低
        if i < num_bands - 1:
            next_band_flat = hsi_data[:, :, i+1].flatten()

            # 确保波段本身和相邻波段不是常数（std > 0），否则相关性可能为 NaN
            current_band_std = np.std(band_flat)
            next_band_std = np.std(next_band_flat)
            if current_band_std > min_std_threshold and next_band_std > 0:
                corr_next = np.corrcoef(band_flat, next_band_flat)[0, 1]
            else:
                corr_next = np.nan # 无法计算相关性
            if np.isnan(corr_next) or corr_next < min_corr_threshold:
                bad_band_indices.add(i)
                # print(f"4 Flagged band {i} ({wavelengths[i]:.1f}nm) (Corr Next: {corr_next:.2f}) - Reason: Low/NaN correlation with next band")
                continue

    return sorted(list(bad_band_indices))


def create_cosine_window(size_y, size_x, blend_width_ratio=0.25):
    y_coords = np.linspace(0, 1, size_y)
    x_coords = np.linspace(0, 1, size_x)

    def cosine_ramp(coord, blend_width):
        if blend_width <= 0 or blend_width >= 0.5: # 确保混合宽度合理
            return 1.0 # 如果混合宽度不合理，就当做没有混合区，全是1
        if coord < blend_width:
            return 0.5 * (1 - np.cos(np.pi * coord / blend_width))
        elif coord > (1 - blend_width):
            return 0.5 * (1 + np.cos(np.pi * (coord - (1 - blend_width)) / blend_width))
        else:
            return 1.0

    y_ramp = np.array([cosine_ramp(c, blend_width_ratio) for c in y_coords])
    x_ramp = np.array([cosine_ramp(c, blend_width_ratio) for c in x_coords])

    window = np.outer(y_ramp, x_ramp)
    return window

def create_blended_mask(mask, rgb, blur_size=100, blur_type='gaussian'):
    """
    对mask边缘进行模糊处理，生成一个用于图像平滑融合的权重mask。

    参数:
    - mask: 输入的二值mask，白色区域(255)为前景，黑色区域(0)为背景。
    - blur_size: 模糊核的大小，必须是正奇数。
    - blur_type: 模糊类型，可选 'gaussian' 或 'median'。

    返回:
    - weight_mask: 0-1之间的浮点数mask，边缘平滑过渡。
    """
    rgb_mask = np.mean(rgb, axis=2)*mask
    blur_size = min(mask.shape)//20
    if blur_size % 2 == 0 or blur_size <= 0:
        blur_size+=1
        # raise ValueError("blur_size 必须是一个正奇数。")

    # 将输入的二值mask转换为浮点数类型
    mask_float = mask.astype(np.float32)

    # 根据选择的类型进行模糊处理
    if blur_type == 'gaussian':
        blurred_mask = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    elif blur_type == 'median':
        # Median blur 在处理mask时效果较好，能更好地保留边缘信息
        # 但需要将mask转换为8位整数类型
        mask_int = mask.astype(np.uint8)
        blurred_mask = cv2.medianBlur(mask_int, blur_size)
    else:
        raise ValueError("不支持的模糊类型，请选择 'gaussian' 或 'median'。")

    return blurred_mask

def outer_ring_highlight(image_rgb, mask, ring_width=10, used_thr=None, thr_quantile=None):
    """
    image_rgb: HxWx3 float, RGB, 值在 [0,1]
    mask: HxW bool 或 {0,1}/{0,255} 二值
    ring_width: 外扩像素宽度（默认10）
    thr: 灰度绝对阈值 (0~1)，与 thr_quantile 二选一
    thr_quantile: 亮度分位数阈值 (0~1)，例如 0.3 表示保留灰度 > 30%分位的像素

    返回:
      ring_mask: 外缘 ring 区域 (bool)
      ring_mask_bright: 外缘且亮度达标的区域 (bool)
      used_thr: 实际使用的阈值 (0~1)
    """
    # --- 1) 规范化 mask 为 bool ---
    m = mask.copy()
    if m.dtype != np.bool_:
        m = m.astype(np.uint8) > 0

    # --- 2) 外扩 ring_width 像素，并减去原 mask 得到 ring ---
    ksize = 2 * ring_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring_mask = np.logical_and(dilated, ~m)

    # --- 5) 筛掉低亮度 ---
    bright = np.mean(image_rgb, axis=-1) > thr_quantile
    ring_mask_bright = np.logical_and(ring_mask, bright)

    return ring_mask, ring_mask_bright, used_thr

# def cosine_window_2d(h, w):
#     """生成2D余弦窗 (hann window)"""
#     y = 0.5 * (1 - np.cos(2 * np.pi * np.arange(h) / (h - 1)))
#     x = 0.5 * (1 - np.cos(2 * np.pi * np.arange(w) / (w - 1)))
#     wy, wx = np.meshgrid(y, x, indexing="ij")
#     return wy * wx

def fill_mask_with_window_means_overlap(xyz_patch, adjusted_xyz_patch, mask, bright_ring,
                                        ratio_scalar=1.0, grid_size=(5,5), overlap=0.25):
    """
    xyz_patch: HxWx3, float
    adjusted_xyz_patch: HxWx3, float (将被修改)
    mask: HxW bool, 需要填充的区域
    bright_ring: HxW bool, 用于计算均值的亮区域
    ratio_scalar: 额外缩放因子
    grid_size: 窗口划分数 (rows, cols)，比如 (5,5)
    overlap: 窗口重叠比例 (0~0.5)，比如 0.25 表示25%重叠
    """
    H, W, _ = xyz_patch.shape
    n_rows, n_cols = grid_size
    win_h = H // n_rows
    win_w = W // n_cols

    step_h = int(win_h * (1 - overlap))
    step_w = int(win_w * (1 - overlap))
    if step_h <= 0 or step_w <= 0:
        raise ValueError("overlap 太大，导致窗口步长 <= 0")

    # 用于累积结果和权重
    accum = np.zeros_like(adjusted_xyz_patch, dtype=np.float64)
    weight = np.zeros((H, W, 1), dtype=np.float64)

    for i in range(0, H, step_h):
        for j in range(0, W, step_w):
            y0, y1 = i, min(i + win_h, H)
            x0, x1 = j, min(j + win_w, W)

            win_mask = mask[y0:y1, x0:x1]
            win_ring = bright_ring[y0:y1, x0:x1]
            win_xyz  = xyz_patch[y0:y1, x0:x1, :]

            if np.any(win_mask):
                if np.any(win_ring):
                    region_vals = win_xyz[win_ring]
                else:
                    region_vals = win_xyz[win_mask]  # fallback

                rgb_xyz_mean = np.mean(region_vals.reshape(-1, 3), axis=0)

                # 指数缩放逻辑
                fill_val = rgb_xyz_mean[np.newaxis, :]**(1.41**(rgb_xyz_mean.mean()+0.2)) * ratio_scalar * 0.85

                # 构造余弦窗 (二维 separable)
                wy = np.hanning(y1 - y0)
                wx = np.hanning(x1 - x0)
                win_wgt = np.outer(wy, wx)[..., None]  # shape (h, w, 1)

                # 只在 mask 区域填充
                win_wgt = win_wgt * win_mask[..., None]

                accum[y0:y1, x0:x1, :] += fill_val * win_wgt
                weight[y0:y1, x0:x1, :] += win_wgt

    # 避免除零
    adjusted_xyz_mask = adjusted_xyz_patch.copy()
    nonzero = weight > 1e-8
    adjusted_xyz_mask[nonzero.squeeze()] = accum[nonzero.squeeze()] / weight[nonzero][:, np.newaxis]

    return adjusted_xyz_mask

def process_image_pair_sliding_window(rgb_path, pkl_path, output_dir,
                                       grid_size=(5,5),
                                       window_size_factor=2/3,
                                       overlap_factor=0.5,
                                       blend_width_ratio=0.25,
                                       min_reasonable_cct=2000,
                                       max_reasonable_cct=15000,
                                       chroma_boost=1.5,  # 色度增强系数，1.0-2.0，越大色彩越鲜艳
                                       tone_mapping_max=1.2,  # 色调映射目标最大值，越大保留更多亮度信息
                                       enable_gamut_extension=False):  # 是否启用扩展色域
    """
    改写版：每个窗口：
      - rgb->xyz 取均值 (rgb_xyz_mean)
      - hsi->xyz->cct 取均值 (hsi_cct_mean)
      - rgb->xyz->cct 取均值 (rgb_cct_mean)
      - mask = (X>1)|(Y>1)|(Z>1)
      - mask区域用 rgb_xyz_mean 替代
      - 非mask区域用 ratio = rgb_cct_mean / hsi_cct_mean 缩放 xyz
      - 在窗口内部使用余弦窗与 mask 做平滑融合
      - 转回 rgb，使用全局余弦窗融合到 final_enhanced_rgb
    """
    bgr_img = cv2.imread(rgb_path)
    if bgr_img is None:
        raise FileNotFoundError(f"RGB image not found at {rgb_path}")
    rgb_img_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img_orig_float = np.asarray(rgb_img_orig, dtype=np.float32) / 255.0

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if 'hsi' not in data:
        raise KeyError(f"pkl file {pkl_path} does not contain 'hsi' key")
    if 'cie_xy' not in data:
        raise KeyError(f"pkl file {pkl_path} does not contain 'cie_xy' key")
    print(f"Data loaded from {pkl_path}")
    hsi_orig = data['hsi']
    cie_xy_orig = np.asarray(data['cie_xy'], dtype=np.float32)
    cie_Y_orig = np.asarray(data['cie_Y'], dtype=np.float32)
    if cie_xy_orig.ndim != 3 or cie_xy_orig.shape[2] != 2:
        raise ValueError(f"'cie_xy' expected shape (H, W, 2), got {cie_xy_orig.shape}")
    # hsi_orig = pickle.load(open(f"HSI_aligned/{pkl_name}.pkl", "rb"))
    wavelengths = np.linspace(380, 980, hsi_orig.shape[2])

    # resize HSI to match RGB spatial dims if needed
    if hsi_orig.shape[0:2] != rgb_img_orig_float.shape[0:2]:
        temp_hsi_resized = np.zeros((rgb_img_orig_float.shape[0], rgb_img_orig_float.shape[1], hsi_orig.shape[2]), dtype=hsi_orig.dtype)
        for i in range(hsi_orig.shape[2]):
            temp_hsi_resized[:,:,i] = cv2.resize(hsi_orig[:,:,i],
                                                 (rgb_img_orig_float.shape[1], rgb_img_orig_float.shape[0]),
                                                 interpolation=cv2.INTER_LINEAR)
        hsi_for_processing = temp_hsi_resized
    else:
        hsi_for_processing = hsi_orig

    if cie_xy_orig.shape[0:2] != rgb_img_orig_float.shape[0:2]:
        cie_xy_resized = np.zeros((rgb_img_orig_float.shape[0], rgb_img_orig_float.shape[1], 2), dtype=cie_xy_orig.dtype)
        for i in range(2):
            cie_xy_resized[:, :, i] = cv2.resize(
                cie_xy_orig[:, :, i],
                (rgb_img_orig_float.shape[1], rgb_img_orig_float.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        cie_xy_for_processing = cie_xy_resized
    else:
        cie_xy_for_processing = cie_xy_orig

    if cie_Y_orig.shape[0:2] != rgb_img_orig_float.shape[0:2]:
        cie_Y_resized = np.zeros((rgb_img_orig_float.shape[0], rgb_img_orig_float.shape[1]), dtype=cie_Y_orig.dtype)
        cie_Y_resized = cv2.resize(
            cie_Y_orig,
            (rgb_img_orig_float.shape[1], rgb_img_orig_float.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        cie_Y_for_processing = cie_Y_resized
    else:
        cie_Y_for_processing = cie_Y_orig

    H, W, _ = rgb_img_orig_float.shape
    final_enhanced_rgb = np.zeros_like(rgb_img_orig_float)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # determine window size and steps (similar to original logic)
    window_h = H // ((grid_size[0] + 1)//2)
    window_w = W // ((grid_size[1] + 1)//2)
    window_h = window_h if window_h % 2 == 0 else window_h - 1
    window_w = window_w if window_w % 2 == 0 else window_w - 1
    step_h = int(window_h * (1 - overlap_factor))
    step_w = int(window_w * (1 - overlap_factor))
    step_h = max(1, step_h)
    step_w = max(1, step_w)
    num_windows_y = (H - window_h) // step_h + 1
    num_windows_x = (W - window_w) // step_w + 1

    # cosine window for smooth blending in each patch
    cosine_window_mask = create_cosine_window(window_h, window_w, blend_width_ratio)

    print(f"Processing with {num_windows_y}x{num_windows_x} sliding windows...")
    print(f"Window size: {window_h}x{window_w}, Step: {step_h}x{step_w}")

    for r in range(num_windows_y):
        for c in range(num_windows_x):
            y_start = r * step_h
            x_start = c * step_w
            y_end = min(y_start + window_h, H)
            x_end = min(x_start + window_w, W)

            current_window_h = y_end - y_start
            current_window_w = x_end - x_start
            if current_window_h <= 0 or current_window_w <= 0:
                continue

            rgb_patch = rgb_img_orig_float[y_start:y_end, x_start:x_end, :]  # Hc x Wc x 3
            hsi_patch = hsi_for_processing[y_start:y_end, x_start:x_end, :]   # Hc x Wc x C
            hsi_xy_patch = cie_xy_for_processing[y_start:y_end, x_start:x_end, :]
            hsi_Y_patch = cie_Y_for_processing[y_start:y_end, x_start:x_end]

            # create appropriate cosine mask for boundary patches
            if current_window_h != window_h or current_window_w != window_w:
                current_cosine_mask = create_cosine_window(current_window_h, current_window_w, blend_width_ratio)
            else:
                current_cosine_mask = cosine_window_mask

            # default: output is original patch (fallback)
            adjusted_rgb_patch = rgb_patch.copy()

            # ensure patch valid
            if hsi_patch.size == 0 or rgb_patch.size == 0:
                print(f"  Warning: Empty patch at ({y_start},{x_start}). Skipping.")
            else:
                # try:
                    # --- 1) RGB patch -> XYZ and its mean ---
                    xyz_patch = rgb_to_xyz(rgb_patch)  # Hc x Wc x 3

                    # --- 2) HSI patch -> XYZ (using all bands) -> CCT map -> mean hsi cct ---
                    hsi_xyz_patch = hsi_to_xyz(hsi_patch, wavelengths)  # Hc x Wc x 3
                    # hsi_cct_map = xyz_to_cct(hsi_xyz_patch)             # Hc x Wc
                    hsi_cct_map = xy_to_cct(hsi_xy_patch)
                    # # keep only reasonable cct values
                    # valid_hsi_cct_mask = (hsi_cct_map >= min_reasonable_cct) & (hsi_cct_map <= max_reasonable_cct)
                    # if np.any(valid_hsi_cct_mask):
                    #     hsi_cct_mean = float(np.mean(hsi_cct_map[valid_hsi_cct_mask]))
                    # else:
                    #     hsi_cct_mean = 0.0

                    # --- 3) RGB patch -> CCT map (from its XYZ) -> mean rgb cct ---
                    rgb_cct_map = xyz_to_cct(xyz_patch)
                    valid_rgb_cct_mask = (rgb_cct_map >= min_reasonable_cct) & (rgb_cct_map <= max_reasonable_cct)
                    if np.any(valid_rgb_cct_mask):
                        rgb_cct_mean = float(np.mean(rgb_cct_map[valid_rgb_cct_mask]))
                    else:
                        rgb_cct_mean = 0.0

                    # --- 4) build mask: where any XYZ channel > 1 (over-range/highlight) ---
                    Xc = xyz_patch[..., 0]
                    Yc = xyz_patch[..., 1]
                    Zc = xyz_patch[..., 2]
                    mask = (Xc >= 1.0) | (Yc >= 1.0) | (Zc >= 1.0)  # boolean mask Hc x Wc

                    # --- 5) compute ratio for non-mask region ---
                    adjusted_xyz_patch = xyz_patch.copy()

                    # if both means valid, compute ratio; else keep ratio = 1 (no change)
                    if (hsi_cct_mean > 0) and (rgb_cct_mean > 0):
                        ratio_scalar = rgb_cct_mean / hsi_cct_mean
                        # clamp ratio to avoid extreme scaling
                        ratio_scalar = float(ratio_scalar)
                        ratio_scalar = float(np.clip(ratio_scalar, 0.1, 10.0))
                        adjusted_xyz_patch *=ratio_scalar
                        # non_mask_idx = ~mask
                        # if np.any(non_mask_idx):
                        #     adjusted_xyz_patch[non_mask_idx] = adjusted_xyz_patch[non_mask_idx] * ratio_scalar
                    else:
                        # no reliable HSI or RGB CCT mean -> skip ratio scaling
                        ratio_scalar = 1.0

                    # --- 6) mask region replacement: set masked pixels to rgb_xyz_mean ---
                    if np.any(mask):
                        # broadcast rgb_xyz_mean into masked pixels
                        adjusted_xyz_mask = adjusted_xyz_patch.copy()
                        rgb_xyz_mean = np.mean(xyz_patch[mask].reshape(-1, 3), axis=0)  # (3,)
                        ring, ring_bright, used_thr = outer_ring_highlight(xyz_patch, mask, ring_width=5, thr_quantile=0.7)
                        # adjusted_xyz_mask[mask] = rgb_xyz_mean[np.newaxis, :]**(1.41**(rgb_xyz_mean.mean()+0.2))*ratio_scalar*0.9
                        adjusted_xyz_mask = fill_mask_with_window_means_overlap(
                            xyz_patch, adjusted_xyz_patch, mask, ring_bright, ratio_scalar, grid_size=(5,5)
                        )
                        blend_weight = create_blended_mask(mask, rgb_patch)
                        adjusted_xyz_patch = adjusted_xyz_mask * blend_weight[:,:,np.newaxis] + adjusted_xyz_patch * (1 - blend_weight)[:,:,np.newaxis]

                    # --- 7) 色域增强：在xyY空间中增强色度 ---
                    if enable_gamut_extension:
                        # 转换到xyY空间
                        rgb_xyY = xyz_to_xyY(adjusted_xyz_patch)
                        hsi_xy_patch = cie_xy_for_processing[y_start:y_end, x_start:x_end, :]
                        hsi_y_values = hsi_Y_patch # hsi_xyz_patch[..., 1:2]
                        hsi_xyY = np.concatenate((hsi_xy_patch, hsi_y_values), axis=-1)

                        # 增强色度（使用配置的chroma_boost参数）
                        enhanced_xyY = enhance_chroma_in_xyY(rgb_xyY, hsi_xyY, chroma_boost=chroma_boost)

                        # 转换回XYZ
                        adjusted_xyz_patch = xyY_to_xyz(enhanced_xyY)

                    # --- 8) 自适应色调映射（替代简单的归一化） ---
                    # 使用Reinhard tone mapping保留色彩，同时压缩动态范围
                    # adjusted_xyz_patch = adaptive_tone_mapping(
                    #     adjusted_xyz_patch, 
                    #     target_max=1.0, # tone_mapping_max, 
                    #     preserve_color=True
                    # )

                    # --- 9) convert merged XYZ back to sRGB for the patch ---
                    if enable_gamut_extension:
                        # 使用扩展色域模式，允许超出标准sRGB范围
                        merged_rgb_patch = xyz_to_rgb(adjusted_xyz_patch, clip_output=False, extended_gamut=True)
                        # 更宽松的clip，保留更多色彩信息
                        adjusted_rgb_patch = merged_rgb_patch
                    else:
                        # 标准模式
                        merged_rgb_patch = xyz_to_rgb(adjusted_xyz_patch, clip_output=True, extended_gamut=False)
                        adjusted_rgb_patch = merged_rgb_patch

                # except Exception as e:
                #     print(f"  Error processing patch ({y_start},{x_start}) to ({y_end},{x_end}): {e}. Using original patch.")
                #     adjusted_rgb_patch = rgb_patch

            # --- accumulate into global canvas with cosine blending ---
            if adjusted_rgb_patch.shape[0:2] != current_cosine_mask.shape:
                # resize cosine mask to patch size if mismatch
                current_cosine_mask_resized = cv2.resize(current_cosine_mask, (adjusted_rgb_patch.shape[1], adjusted_rgb_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                current_cosine_mask_resized = current_cosine_mask

            final_enhanced_rgb[y_start:y_end, x_start:x_end, :] += adjusted_rgb_patch * current_cosine_mask_resized[:, :, np.newaxis]
            weight_map[y_start:y_end, x_start:x_end] += current_cosine_mask_resized

    # Normalize by weight map
    weight_map[weight_map == 0] = 1.0
    final_enhanced_rgb = final_enhanced_rgb / weight_map[:, :, np.newaxis]
    
    # --- 全局色调映射：保留扩展色域，只在必要时压缩 ---
    # 检查是否有超出[0,1]范围的值
    max_val = np.max(final_enhanced_rgb)
    if max_val > 1.0:
        # 应用软压缩，保留色彩信息
        # 使用分段函数：[0,1]保持不变，>1的部分进行对数压缩
        final_enhanced_rgb = np.where(
            final_enhanced_rgb <= 1.0,
            final_enhanced_rgb,
            1.0 + np.log(final_enhanced_rgb) / np.log(max_val + 1)
        )
    
    # save
    base_name = os.path.splitext(os.path.basename(rgb_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.png")
    final_rgb_to_save = np.clip(final_enhanced_rgb, 0.0, 1.0)
    Image.fromarray((final_rgb_to_save * 255.0).astype(np.uint8)).save(output_path)
    print(f"✅ Enhanced image saved: {output_path}")

# --------------------------
# 🔥 批量处理 - 调用新的滑动窗口函数
# --------------------------
def batch_process_sliding_window(folder_rgb, folder_hsi, output_dir,
                                 chroma_boost=1.5,  # 色度增强系数
                                 tone_mapping_max=1.2,  # 色调映射最大值
                                 enable_gamut_extension=True):  # 是否启用扩展色域
    """
    批量处理RGB和HSI图像对
    
    参数：
    - chroma_boost: 色度增强系数，建议范围1.0-2.0
      * 1.0 = 不增强（保持原始色彩）
      * 1.3-1.5 = 适中增强（推荐）
      * 1.8-2.0 = 强烈增强（色彩非常鲜艳）
    
    - tone_mapping_max: 色调映射目标最大亮度，建议范围0.95-1.5
      * 0.95 = 保守（不会超出标准范围）
      * 1.2 = 适中（推荐，允许一定的HDR效果）
      * 1.5 = 激进（保留更多高光细节）
    
    - enable_gamut_extension: 是否启用扩展色域
      * True = 允许超出sRGB色域（更广色域）
      * False = 限制在sRGB色域内（标准）
    """
    file_list = os.listdir(folder_rgb)
    print(f"Found {len(file_list)} files. Processing with:")
    print(f"  - Chroma boost: {chroma_boost}")
    print(f"  - Tone mapping max: {tone_mapping_max}")
    print(f"  - Gamut extension: {enable_gamut_extension}")
    
    for file in file_list[43:]:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            rgb_path = os.path.join(folder_rgb, file)
            pkl_name = os.path.splitext(file)[0].replace('_out', '')
            pkl_path = os.path.join(folder_hsi, f"{pkl_name}.pkl")
            if os.path.exists(pkl_path):
                # try:
                    process_image_pair_sliding_window(
                        rgb_path, pkl_path, output_dir,
                        chroma_boost=chroma_boost,
                        tone_mapping_max=tone_mapping_max,
                        enable_gamut_extension=enable_gamut_extension
                    )
                # except Exception as e:
                #     print(f"❌ Failed processing {file}: {e}")
            else:
                print(f"⚠️ No HSI file for {file}")

if __name__ == "__main__":
    folder_rgb = "exp_results/RealESRGAN_enhanced" # "/data2/dataset/HSI_dataset/Glasses_img"
    folder_hsi =  'HSI_dataset/HSI_data'    # "HSI_aligned" #
    output_dir = "new_out/RealESRGAN_HSI" # 新的输出目录，区分版本
    
    # ====== 色域增强配置 ======
    # 根据您的需求调整以下参数：
    
    # 方案1：适中增强（推荐作为起点）
    batch_process_sliding_window(
        folder_rgb, folder_hsi, output_dir,
        chroma_boost=1.5,           # 适度增强色彩饱和度
        tone_mapping_max=1.2,       # 允许一定HDR效果
        enable_gamut_extension=False # 启用扩展色域
    )
    
    # 方案2：激进增强（最大色域，适合对比测试）
    # batch_process_sliding_window(
    #     folder_rgb, folder_hsi, output_dir,
    #     chroma_boost=1.8,           # 大幅增强色彩
    #     tone_mapping_max=1.5,       # 保留更多高光
    #     enable_gamut_extension=True
    # )
    
    # 方案3：保守模式（接近原始效果）
    # batch_process_sliding_window(
    #     folder_rgb, folder_hsi, output_dir,
    #     chroma_boost=1.2,
    #     tone_mapping_max=0.95,
    #     enable_gamut_extension=False  # 限制在sRGB色域
    # )