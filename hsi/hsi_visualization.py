
"""
HSI 数据可视化与导出工具

该脚本提供独立的函数，用于处理从高光谱（HSI）数据文件中提取的信息，
并将其转换为可用的RGB图像和分析图表。

主要功能:
1.  从 HSI 数据中的 'cie_XYZ' 数据转换并生成 RGB 图像。
2.  从 HSI 数据中提取预计算的 'rgb' 图像。
3.  为上述两种RGB结果生成并保存 CIE xy 和 CIELAB a*b* 色域图。

使用方法:
-   在脚本底部的 `main` 函数中，修改 `HSI_FILE_PATH` 指向你的 `.pkl` 文件。
-   运行脚本，生成的图像和图表将保存在 `OUTPUT_DIR` 指定的目录中。
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from colour.models import RGB_COLOURSPACE_sRGB
from PIL import Image

# --- 色彩空间转换与绘图辅助函数 ---
# (从 hsi_pipeline_advanced_correction.py 移植以确保脚本独立性)

def rgb_to_xyz(rgb_img: np.ndarray) -> np.ndarray:
    """sRGB (0-1) to XYZ. Input must be float and in [0, 1] range."""
    rgb_to_xyz_matrix = RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ
    
    def linearize_srgb(rgb: np.ndarray) -> np.ndarray:
        mask = rgb <= 0.04045
        linear_rgb = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return linear_rgb
        
    linear_rgb = linearize_srgb(np.clip(rgb_img, 0, 1))
    xyz_img = np.tensordot(linear_rgb, rgb_to_xyz_matrix, axes=([2], [1]))
    return xyz_img

def xyz_to_rgb(xyz_img: np.ndarray) -> np.ndarray:
    """XYZ to sRGB (0-1)."""
    xyz_to_rgb_matrix = RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB
    linear_rgb = np.tensordot(xyz_img, xyz_to_rgb_matrix, axes=([2], [1]))
    
    def gamma_encode(linear: np.ndarray) -> np.ndarray:
        linear = np.clip(linear, 0, None)
        mask = linear <= 0.0031308
        srgb = np.where(mask, 12.92 * linear, 1.055 * (linear ** (1/2.4)) - 0.055)
        return srgb
        
    rgb_img = gamma_encode(linear_rgb)
    return np.clip(rgb_img, 0, 1)

def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """XYZ to CIELAB."""
    ref_white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32) # D65
    
    def f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4.0 / 29.0))

    xyz_norm = xyz / ref_white
    fx, fy, fz = f(xyz_norm[..., 0]), f(xyz_norm[..., 1]), f(xyz_norm[..., 2])
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1)

def xyz_to_xy(xyz_img: np.ndarray) -> np.ndarray:
    """XYZ to xy."""
    X, Y, Z = xyz_img[..., 0], xyz_img[..., 1], xyz_img[..., 2]
    sum_xyz = X + Y + Z
    valid_mask = sum_xyz > 1e-8
    x = np.full_like(X, np.nan)
    y = np.full_like(Y, np.nan)
    x[valid_mask] = X[valid_mask] / sum_xyz[valid_mask]
    y[valid_mask] = Y[valid_mask] / sum_xyz[valid_mask]
    return np.stack([x, y], axis=-1)

def xyY_to_xyz(xyY_img: np.ndarray) -> np.ndarray:
    """xyY to XYZ."""
    x, y, Y = xyY_img[..., 0], xyY_img[..., 1], xyY_img[..., 2]
    # 为避免除以零，对y值进行安全处理
    y_safe = np.where(y > 1e-12, y, 1e-12)
    X = (Y / y_safe) * x
    Z = (Y / y_safe) * (1 - x - y)
    return np.stack([X, Y, Z], axis=-1)

def visualize_gamut_plots(
    xyz_data: np.ndarray, 
    rgb_data: np.ndarray, 
    plot_prefix: str, 
    output_dir: Path, 
    downsample_factor: int = 8
):
    """
    绘制并保存给定色彩数据的Lab和xy空间投影图。
    """
    print(f"  Visualizing gamut for '{plot_prefix}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if downsample_factor > 1:
        xyz_data = xyz_data[::downsample_factor, ::downsample_factor, :]
        rgb_data = rgb_data[::downsample_factor, ::downsample_factor, :]

    pixel_count = xyz_data.shape[0] * xyz_data.shape[1]
    xyz_flat = xyz_data.reshape(pixel_count, 3)
    rgb_flat_float = rgb_data.reshape(pixel_count, 3).astype(np.float32) / 255.0

    # --- 1. 绘制 a*b* 图 ---
    lab_flat = xyz_to_lab(xyz_flat)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(lab_flat[:, 1], lab_flat[:, 2], c=rgb_flat_float, s=5, alpha=0.7, edgecolors='none')
    plt.xlim(-128, 128); plt.ylim(-128, 128)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("a* (Green-Red)"); plt.ylabel("b* (Blue-Yellow)")
    plt.title(f"Gamut in CIELAB a*b* plane ({plot_prefix})")
    plt.gca().set_aspect('equal', adjustable='box')
    
    lab_output_path = output_dir / f"{plot_prefix}_lab.png"
    plt.savefig(lab_output_path, dpi=150)
    plt.close()
    
    # --- 2. 绘制 xy 色度图 ---
    xy_flat = xyz_to_xy(xyz_flat)
    
    plt.figure(figsize=(8, 9))
    plt.scatter(xy_flat[:, 0], xy_flat[:, 1], c=rgb_flat_float, s=5, alpha=0.7, edgecolors='none')
    plt.xlim(0, 0.8); plt.ylim(0, 0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("CIE x"); plt.ylabel("CIE y")
    plt.title(f"Gamut in CIE xy Chromaticity ({plot_prefix})")
    plt.gca().set_aspect('equal', adjustable='box')

    xy_output_path = output_dir / f"{plot_prefix}_xy.png"
    plt.savefig(xy_output_path, dpi=150)
    plt.close()
    print(f"  Saved gamut plots to {output_dir}")

# --- 核心转换函数 ---

def convert_hsi_spectral_to_rgb(hsi_data: dict) -> np.ndarray:
    """
    通过选择最接近红、绿、蓝波长的波段，将HSI光谱数据直接转换为RGB图像。
    
    这是一种用于快速可视化的直接方法，但色彩还原不一定精确。
    假设 hsi_data 字典包含 'spectral_data' 和 'wavelengths'。
    """
    print("Converting HSI 'spectral_data' to RGB by band selection...")
    if 'hsi' not in hsi_data or 'chdef' not in hsi_data:
        raise ValueError("HSI data must contain 'hsi' and 'chdef' keys.")
    
    spectral_cube = np.asarray(hsi_data['hsi'], dtype=np.float32)
    wavelengths = np.asarray(hsi_data['chdef'], dtype=np.float32)
    
    # 定义目标波长 (nm)
    target_wl = {'r': 640, 'g': 550, 'b': 460}
    
    # 找到最接近目标波长的波段索引
    band_indices = {}
    for channel, wl in target_wl.items():
        band_indices[channel] = np.argmin(np.abs(wavelengths - wl))
        print(f"  Channel {channel.upper()}: Using band at {wavelengths[band_indices[channel]]:.1f} nm (target: {wl} nm)")

    # 提取对应波段
    r_band = spectral_cube[:, :, band_indices['r']]
    g_band = spectral_cube[:, :, band_indices['g']]
    b_band = spectral_cube[:, :, band_indices['b']]
    
    # 对每个通道独立进行归一化，以保留各自的对比度
    def normalize_channel(channel: np.ndarray) -> np.ndarray:
        # 为避免极端值影响，采用百分位数进行裁剪
        min_val = np.percentile(channel, 1)
        max_val = np.percentile(channel, 99)
        
        # 处理特殊情况（例如，通道数值恒定）
        if max_val <= min_val:
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val <= min_val:
                return np.zeros_like(channel, dtype=np.float32)

        # 归一化到 [0, 1] 范围
        normalized_channel = (channel - min_val) / (max_val - min_val)
        return np.clip(normalized_channel, 0, 1)

    r_normalized = normalize_channel(r_band)
    g_normalized = normalize_channel(g_band)
    b_normalized = normalize_channel(b_band)
    
    # 堆叠成RGB图像
    rgb_normalized = np.stack([r_normalized, g_normalized, b_normalized], axis=-1)
    
    # 转换为 uint8
    rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
    
    return rgb_uint8

def convert_hsi_xyz_to_rgb(hsi_data: dict) -> np.ndarray:
    """
    从 HSI 数据字典中提取 'cie_XYZ' 数据并将其转换为 sRGB 图像。
    """
    print("Converting HSI 'cie_XYZ' to RGB image...")
    if 'cie_XYZ' not in hsi_data:
        raise ValueError("HSI data dictionary must contain the key 'cie_XYZ'.")
    
    hsi_xyz = np.asarray(hsi_data['cie_XYZ'], dtype=np.float32)
    
    # 将XYZ转换为范围在[0, 1]的浮点型RGB图像
    rgb_float = xyz_to_rgb(hsi_xyz)
    
    # 转换为范围在[0, 255]的8位无符号整型图像
    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    
    return rgb_uint8

def convert_hsi_xyY_to_rgb(hsi_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    通过拼接 HSI 数据中的 'cie_xy' 和 'cie_Y' (或 'cie_XYZ'中的Y通道)
    来重建 xyY 图像，然后转换为 sRGB 图像。
    返回转换后的RGB图像和中间过程的XYZ图像。
    """
    print("Reconstructing from 'cie_xy' and 'Y' to RGB image...")
    if 'cie_xy' not in hsi_data:
        raise ValueError("HSI data must contain 'cie_xy' key.")

    hsi_xy = np.asarray(hsi_data['cie_xy'], dtype=np.float32)
    
    if 'cie_Y' in hsi_data:
        hsi_Y = np.asarray(hsi_data['cie_Y'], dtype=np.float32)
    elif 'cie_XYZ' in hsi_data:
        print("  'cie_Y' not found, using Y channel from 'cie_XYZ'.")
        hsi_Y = np.asarray(hsi_data['cie_XYZ'], dtype=np.float32)[..., 1]
    else:
        raise ValueError("HSI data must contain either 'cie_Y' or 'cie_XYZ' key.")

    # 确保 Y 只有 HxW 维度
    if hsi_Y.ndim == 3 and hsi_Y.shape[-1] == 1:
        hsi_Y = np.squeeze(hsi_Y, axis=-1)

    # 拼接成 xyY 图像
    hsi_xyY = np.dstack((hsi_xy, hsi_Y))
    
    # 转换: xyY -> XYZ -> RGB
    hsi_xyz = xyY_to_xyz(hsi_xyY)
    rgb_float = xyz_to_rgb(hsi_xyz)
    
    # 转换为 uint8
    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    
    return rgb_uint8, hsi_xyz

def get_hsi_precomputed_rgb(hsi_data: dict) -> np.ndarray:
    """
    从 HSI 数据字典中直接提取预计算的 'rgb' 图像。
    这通常是HSI原始光谱数据到RGB的一种直接映射。
    """
    print("Extracting pre-computed 'rgb' from HSI data...")
    if 'rgb' not in hsi_data:
        raise ValueError("HSI data dictionary must contain the key 'rgb'.")
        
    rgb_uint8 = np.asarray(hsi_data['rgb'], dtype=np.uint8)
    # 确保图像是RGB顺序，如果不是则进行转换（例如，从BGR）
    # 假设hsi_data['rgb']已经是RGB格式
    return rgb_uint8

def visualize_nir_channel(hsi_data: dict, output_dir: Path, file_stem: str):
    """
    提取、处理并保存 HSI 数据中的近红外（NIR）通道作为灰度图像。
    """
    print("Visualizing NIR channel...")
    if 'nir' not in hsi_data:
        raise ValueError("HSI data dictionary must contain the key 'nir'.")
        
    nir_data = np.asarray(hsi_data['nir'], dtype=np.float32)
    
    # 对单通道进行归一化以获得良好的可视化对比度
    min_val = np.percentile(nir_data, 1)
    max_val = np.percentile(nir_data, 99)
    
    if max_val <= min_val:
        min_val = np.min(nir_data)
        max_val = np.max(nir_data)
        if max_val <= min_val:
            nir_uint8 = np.zeros_like(nir_data, dtype=np.uint8)
        else:
            nir_normalized = (nir_data - min_val) / (max_val - min_val)
            nir_uint8 = (np.clip(nir_normalized, 0, 1) * 255).astype(np.uint8)
    else:
        nir_normalized = (nir_data - min_val) / (max_val - min_val)
        nir_uint8 = (np.clip(nir_normalized, 0, 1) * 255).astype(np.uint8)
        
    # 保存为灰度图像
    save_path = output_dir / f"{file_stem}_nir_channel.png"
    # 使用 'L' 模式来保存为灰度图
    Image.fromarray(nir_uint8, 'L').save(save_path)
    print(f"Saved NIR channel image to: {save_path}")

def main():
    """
    主执行函数
    """
    # --- 用户配置 ---
    # 请修改此路径为您要处理的高光谱数据文件
    HSI_FILE_PATH = Path('HSI_dataset/HSI_data/139.pkl')
    # 输出结果将保存在此目录
    OUTPUT_DIR = Path('new_out/hsi_visualization_output')
    # -----------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not HSI_FILE_PATH.exists():
        print(f"Error: HSI data file not found at '{HSI_FILE_PATH}'")
        return

    print(f"Loading HSI data from: {HSI_FILE_PATH}")
    with HSI_FILE_PATH.open("rb") as f:
        hsi_data = pickle.load(f)
        
    file_stem = HSI_FILE_PATH.stem

    # --- 1. 从 'cie_XYZ' 转换 ---
    try:
        rgb_from_xyz = convert_hsi_xyz_to_rgb(hsi_data)
        
        # 保存图像
        save_path_xyz = OUTPUT_DIR / f"{file_stem}_from_xyz.png"
        Image.fromarray(rgb_from_xyz).save(save_path_xyz)
        print(f"Saved RGB image from XYZ to: {save_path_xyz}")

        # 生成并保存色域图
        visualize_gamut_plots(
            xyz_data=np.asarray(hsi_data['cie_XYZ'], dtype=np.float32),
            rgb_data=rgb_from_xyz,
            plot_prefix=f"{file_stem}_from_xyz",
            output_dir=OUTPUT_DIR / "gamut_visuals"
        )
    except ValueError as e:
        print(f"Could not convert from 'cie_XYZ': {e}")

    # --- 2. 提取预计算的 'rgb' ---
    try:
        rgb_precomputed = get_hsi_precomputed_rgb(hsi_data)

        # 保存图像
        save_path_precomputed = OUTPUT_DIR / f"{file_stem}_precomputed_rgb.png"
        Image.fromarray(rgb_precomputed).save(save_path_precomputed)
        print(f"Saved pre-computed RGB image to: {save_path_precomputed}")

        # 生成并保存色域图
        # 注意：我们仍然使用'cie_XYZ'作为色域图的坐标基础，因为它是更标准的色彩表示
        if 'cie_XYZ' in hsi_data:
            visualize_gamut_plots(
                xyz_data=np.asarray(hsi_data['cie_XYZ'], dtype=np.float32),
                rgb_data=rgb_precomputed,
                plot_prefix=f"{file_stem}_precomputed_rgb",
                output_dir=OUTPUT_DIR / "gamut_visuals"
            )
    except ValueError as e:
        print(f"Could not extract pre-computed 'rgb': {e}")

    # --- 3. 从光谱数据直接转换 ---
    try:
        rgb_from_spectral = convert_hsi_spectral_to_rgb(hsi_data)
        
        # 保存图像
        save_path_spectral = OUTPUT_DIR / f"{file_stem}_from_spectral.png"
        Image.fromarray(rgb_from_spectral).save(save_path_spectral)
        print(f"Saved RGB image from spectral data to: {save_path_spectral}")
        
        # 为了生成色域图，我们需要将这个RGB图像转换回XYZ空间
        rgb_float_spectral = rgb_from_spectral.astype(np.float32) / 255.0
        xyz_from_spectral = rgb_to_xyz(rgb_float_spectral)
        
        # 生成并保存色域图
        visualize_gamut_plots(
            xyz_data=xyz_from_spectral,
            rgb_data=rgb_from_spectral,
            plot_prefix=f"{file_stem}_from_spectral",
            output_dir=OUTPUT_DIR / "gamut_visuals"
        )
    except ValueError as e:
        print(f"Could not convert from spectral data: {e}")
        
    # --- 4. 从 'cie_xy' 和 'cie_Y' 重建 ---
    try:
        rgb_from_xyY, xyz_from_xyY = convert_hsi_xyY_to_rgb(hsi_data)
        
        # 保存图像
        save_path_xyY = OUTPUT_DIR / f"{file_stem}_from_xyY.png"
        Image.fromarray(rgb_from_xyY).save(save_path_xyY)
        print(f"Saved RGB image from xyY to: {save_path_xyY}")
        
        # 生成并保存色域图
        visualize_gamut_plots(
            xyz_data=xyz_from_xyY,
            rgb_data=rgb_from_xyY,
            plot_prefix=f"{file_stem}_from_xyY",
            output_dir=OUTPUT_DIR / "gamut_visuals"
        )
    except ValueError as e:
        print(f"Could not convert from xyY data: {e}")

    # --- 5. 可视化 NIR 通道 ---
    try:
        visualize_nir_channel(hsi_data, OUTPUT_DIR, file_stem)
    except ValueError as e:
        print(f"Could not visualize NIR channel: {e}")

    print("\nVisualization process finished.")

if __name__ == "__main__":
    main()
