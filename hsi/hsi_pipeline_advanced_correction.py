"""
HSI高级色彩校正与色域增强工具 (Advanced Correction Pipeline)

该脚本实现一个两步走的色彩校正流程，旨在利用高光谱(HSI)数据的
丰富信息来修复和增强常规RGB图像的色彩表现。

流程核心:
1.  **白平衡校正 (White Balance Correction)**:
    -   利用RGB图像中的中性色区域（白色/灰色/高光）来精确估计场景光源的色温。
    -   将该色温与从HSI数据中相同区域计算出的“地面真实”色温进行对比。
    -   计算一个校正比例，对整个RGB图像的色偏进行校正，确保白点准确。

2.  **引导式色域拓宽 (Guided Gamut Enhancement)**:
    -   在xyY色彩空间中操作，该空间将色度(xy)与亮度(Y)完美分离。
    -   以HSI数据提供的xy色度值为“目标”，将经过白平衡校正的RGB图像的
        每个像素的色度向目标色度进行引导式插值。
    -   在增强色度饱和度和准确性的同时，保持原始图像的亮度和对比度结构不变。

3.  **Rec.2020 输出 (可选)**:
    -   输入 Glasses 图像为 sRGB，色域有限（尤其绿色）。可利用高光谱信息将
        色彩映射到 Rec.2020 空间：target_gamut="Rec.2020" 时放宽色度约束至
        Rec.2020 三角形，output_colourspace="Rec.2020" 时最终以 Rec.2020 编码保存，
        绿色等饱和色可更好保留。

该脚本被设计为模块化，允许用户独立启用或禁用上述步骤，并可调整
色域拓宽的强度与输出色彩空间。

使用方法:
-   在脚本末尾的 `if __name__ == "__main__":` 部分修改输入输出路径。
-   调整 `run_advanced_pipeline` 函数调用中的参数来控制流程，例如:
    -   `do_white_balance`: True/False，是否执行白平衡。
    -   `do_chroma_enhance`: True/False，是否执行色域拓宽。
    -   `enhancement_strength`: 0.0-1.0，色域拓宽的强度。
"""

import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from colour.models import RGB_COLOURSPACE_sRGB
from scipy.interpolate import interp1d
from skimage import img_as_float32
import matplotlib.pyplot as plt
import colour

# --- 辅助函数与色彩空间转换 ---

def xyz_to_xy(xyz_img):
    """XYZ to xy."""
    X, Y, Z = xyz_img[..., 0], xyz_img[..., 1], xyz_img[..., 2]
    sum_xyz = X + Y + Z
    valid_mask = sum_xyz > 1e-8
    x = np.full_like(X, np.nan)
    y = np.full_like(Y, np.nan)
    x[valid_mask] = X[valid_mask] / sum_xyz[valid_mask]
    y[valid_mask] = Y[valid_mask] / sum_xyz[valid_mask]
    return np.stack([x, y], axis=-1)

def visualize_gamut_plots(
    xyz_data, 
    rgb_data, 
    plot_prefix: str, 
    output_dir: Path, 
    downsample_factor: int = 8
):
    """
    一个封闭的函数，用于绘制并保存给定色彩数据的Lab和xy空间投影图。
    """
    print(f"  Visualizing gamut for '{plot_prefix}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Downsample for performance
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
    
    # --- 2. 绘制 xy 色度图（点的位置=颜色真实 xy，未限定在 sRGB）---
    xy_flat = xyz_to_xy(xyz_flat)
    valid = np.isfinite(xy_flat[:, 0]) & np.isfinite(xy_flat[:, 1])
    xy_plot = xy_flat[valid]
    rgb_plot = rgb_flat_float[valid]

    plt.figure(figsize=(8, 9))
    plt.scatter(xy_plot[:, 0], xy_plot[:, 1], c=rgb_plot, s=5, alpha=0.7, edgecolors='none')
    # 画出 sRGB 与 Rec.2020 色域三角形，便于看分布是否在 Rec.2020 内
    srgb = GAMUT_PRIMARIES["sRGB"]
    rec = GAMUT_PRIMARIES["Rec.2020"]
    for name, prim, style in [
        ("sRGB", srgb, "b--"),
        ("Rec.2020", rec, "g-"),
    ]:
        r, g, b = prim["r"], prim["g"], prim["b"]
        tri_x = [r[0], g[0], b[0], r[0]]
        tri_y = [r[1], g[1], b[1], r[1]]
        plt.plot(tri_x, tri_y, style, linewidth=1.5, label=name)
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("CIE x")
    plt.ylabel("CIE y")
    plt.title(f"Gamut in CIE xy ({plot_prefix}) — points = actual xy")
    plt.legend(loc="upper right", fontsize=8)
    plt.gca().set_aspect('equal', adjustable='box')

    xy_output_path = output_dir / f"{plot_prefix}_xy.png"
    plt.savefig(xy_output_path, dpi=150)
    plt.close()

def rgb_to_xyz(rgb_img):
    """sRGB (0-1) to XYZ. Input must be float and in [0, 1] range."""
    rgb_to_xyz_matrix = RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ
    
    def linearize_srgb(rgb):
        mask = rgb <= 0.04045
        linear_rgb = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return linear_rgb
        
    linear_rgb = linearize_srgb(np.clip(rgb_img, 0, 1))
    xyz_img = np.tensordot(linear_rgb, rgb_to_xyz_matrix, axes=([2], [1]))
    return xyz_img

def xyz_to_rgb(xyz_img):
    """XYZ to sRGB (0-1)."""
    xyz_to_rgb_matrix = RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB
    linear_rgb = np.tensordot(xyz_img, xyz_to_rgb_matrix, axes=([2], [1]))
    
    def gamma_encode(linear):
        # Clip negative values that can appear from gamut mapping
        linear = np.clip(linear, 0, None)
        mask = linear <= 0.0031308
        srgb = np.where(mask, 12.92 * linear, 1.055 * (linear ** (1/2.4)) - 0.055)
        return srgb
        
    rgb_img = gamma_encode(linear_rgb)
    return np.clip(rgb_img, 0, 1)


# Rec.2020 (BT.2020) 色域更大，尤其绿色，适合从高光谱映射后的输出
# XYZ -> Linear Rec.2020 矩阵 (D65)
_XYZ_TO_REC2020 = np.array([
    [1.71665119, -0.35567078, -0.25336628],
    [-0.66668435, 1.61648124, 0.01576855],
    [0.01763986, -0.04277061, 0.94210312],
], dtype=np.float32)


def _oetf_rec2020(linear):
    """Rec.2020 OETF (linear -> non-linear), 10-bit 参数."""
    alpha, beta = 1.09929682680944, 0.018053968510807
    linear = np.clip(linear, 0, None)
    mask = linear < beta
    return np.where(
        mask,
        linear * 4.5,
        1.099 * (linear ** 0.45) - 0.099
    )


def xyz_to_rec2020(xyz_img):
    """XYZ to Rec.2020 RGB (0-1)。输出为 Rec.2020 编码，色域大于 sRGB，绿色表现更好。"""
    linear = np.tensordot(xyz_img, _XYZ_TO_REC2020.T, axes=([2], [0]))
    rec2020 = _oetf_rec2020(np.clip(linear, 0, None))
    return np.clip(rec2020, 0, 1)

def xyz_to_xyY(xyz_img):
    """XYZ to xyY."""
    X, Y, Z = xyz_img[..., 0], xyz_img[..., 1], xyz_img[..., 2]
    sum_xyz = X + Y + Z + 1e-12
    x = X / sum_xyz
    y = Y / sum_xyz
    return np.stack([x, y, Y], axis=-1)

def xyY_to_xyz(xyY_img):
    """xyY to XYZ."""
    x, y, Y = xyY_img[..., 0], xyY_img[..., 1], xyY_img[..., 2]
    y_safe = np.where(y > 1e-12, y, 1e-12)
    X = (Y / y_safe) * x
    Z = (Y / y_safe) * (1 - x - y)
    return np.stack([X, Y, Z], axis=-1)
    
def xyz_to_lab(xyz):
    """XYZ to CIELAB."""
    ref_white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32) # D65
    
    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4.0 / 29.0))

    xyz_norm = xyz / ref_white
    
    fx, fy, fz = f(xyz_norm[..., 0]), f(xyz_norm[..., 1]), f(xyz_norm[..., 2])
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1)

def xy_to_cct(xy_img, safety_margin=1e-3):
    """
    Robust CCT estimation from xy, returning np.nan for invalid points.
    """
    if xy_img.ndim < 2 or xy_img.shape[-1] < 2:
        raise ValueError("Input must be at least 2D with xy in the last dimension.")
        
    x, y = xy_img[..., 0], xy_img[..., 1]
    cct = np.full(x.shape, np.nan, dtype=np.float32)
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    valid_mask &= (np.abs(0.1858 - y) > safety_margin)
    
    if np.any(valid_mask):
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        n = (x_valid - 0.3320) / (0.1858 - y_valid)
        cct_values = 437 * (n**3) + 3601 * (n**2) + 6861 * n + 5517
        cct[valid_mask] = cct_values
        
    return cct

# --- Core Correction Steps ---

def correct_white_balance_gain(rgb_xyz, hsi_xyz, l_thresh=60, sat_thresh=25):
    """
    Performs a more robust white balance by directly aligning the XYZ values of neutral regions.
    """
    print("Step 1 (Upgraded): Correcting white balance via direct XYZ gain...")
    rgb_lab = xyz_to_lab(rgb_xyz)
    saturation = np.sqrt(rgb_lab[..., 1]**2 + rgb_lab[..., 2]**2)
    neutral_mask = (rgb_lab[..., 0] > l_thresh) & (saturation < sat_thresh)
    
    if not np.any(neutral_mask):
        print("  Warning: No neutral regions found. Skipping white balance.")
        return rgb_xyz

    hsi_white_point = np.mean(hsi_xyz[neutral_mask], axis=0)
    rgb_white_point = np.mean(rgb_xyz[neutral_mask], axis=0)

    # Avoid division by zero
    if np.any(rgb_white_point < 1e-6):
        print("  Warning: RGB white point is near zero. Skipping white balance.")
        return rgb_xyz

    correction_gains = hsi_white_point / rgb_white_point
    # Clamp gains to prevent extreme corrections
    correction_gains = np.clip(correction_gains, 0.5, 2.0)
    print(f"  White Points -> HSI: {hsi_white_point}, RGB: {rgb_white_point}")
    print(f"  Calculated XYZ Gains: {correction_gains}")

    balanced_xyz = rgb_xyz * correction_gains
    
    # 1. Temporarily convert to Linear RGB to check strictly
    # (Since clipping happens in RGB space, not XYZ)
    balanced_rgb_linear = np.tensordot(balanced_xyz, RGB_COLOURSPACE_sRGB.matrix_XYZ_to_RGB, axes=([2], [1]))
    
    # Check for clipping in the linear domain
    # We want to preserve the brightest highlight.
    # Using 99.9th percentile is safer than max to avoid hot pixels making everything too dark.
    max_val = np.percentile(balanced_rgb_linear, 99.9)
    
    if max_val > 1.0:
        print(f"  Anti-Clipping: Scaling down intensity by {1.0/max_val:.3f} (Max val: {max_val:.3f})")
        balanced_xyz = balanced_xyz / max_val
        
    return balanced_xyz


def enhance_by_gamut_expansion(
    xyz_img,
    hsi_xy_map,
    sum_mask,
    global_strength=0.8,
    target_gamut="Rec.2020",
):
    """
    利用 HSI xy 图做色域拓宽。target_gamut 为 "Rec.2020" 时允许色度落入更大范围（尤其绿色），
    配合输出 Rec.2020 可更好保留高光谱信息；"sRGB" 则保持原约束。
    """
    print(f"Step 2 (Upgraded v3): Enhancing chroma via HSI cube analysis (target_gamut={target_gamut})...")
    if global_strength <= 0:
        return xyz_img

    primaries = GAMUT_PRIMARIES.get(target_gamut, GAMUT_PRIMARIES["Rec.2020"])
    r, g, b = primaries["r"], primaries["g"], primaries["b"]

    # --- 1. Define the anchor point for saturation (D65 white point) ---
    white_point_xy = np.array([0.3127, 0.3290], dtype=np.float32)

    # --- 2. Calculate saturation distances and create final enhancement mask ---
    rgb_xyY = xyz_to_xyY(xyz_img)
    rgb_xy = rgb_xyY[..., :2]
    
    dist_rgb = np.linalg.norm(rgb_xy - white_point_xy, axis=-1)
    dist_hsi = np.linalg.norm(hsi_xy_map - white_point_xy, axis=-1)
    dist_hsi_rgb = np.linalg.norm(hsi_xy_map - rgb_xy, axis=-1)
    
    # --- Constraint: Halfway point must be inside target gamut triangle ---
    half_points = rgb_xy + 0.5 * (hsi_xy_map - rgb_xy)
    hsi_rgb_mask = _point_in_triangle_xy(half_points, r, g, b)

    valid_hsi_xy_mask = ~np.isnan(dist_hsi)
    
    # --- New Constraint: Hue Angle Consistency ---
    # We want to avoid shifting hues too drastically (e.g., red to green).
    # We calculate the cosine similarity between the vector (White -> RGB) and (White -> HSI).
    
    vec_rgb = rgb_xy - white_point_xy
    vec_hsi = hsi_xy_map - white_point_xy
    
    # Normalize vectors (avoid division by zero)
    norm_rgb = np.linalg.norm(vec_rgb, axis=-1)
    norm_hsi = np.linalg.norm(vec_hsi, axis=-1)
    
    # Create a mask for valid vectors (length > 0)
    valid_vec_mask = (norm_rgb > 1e-6) & (norm_hsi > 1e-6)
    
    # Calculate cosine similarity
    # cos_theta = (A . B) / (|A| * |B|)
    dot_product = np.sum(vec_rgb * vec_hsi, axis=-1)
    cos_similarity = np.zeros_like(dot_product)
    
    # Only calculate where vectors are valid
    cos_similarity[valid_vec_mask] = dot_product[valid_vec_mask] / (norm_rgb[valid_vec_mask] * norm_hsi[valid_vec_mask])
    
    # --- Weighting based on Angle Consistency ---
    # Instead of a hard threshold, we scale the enhancement strength based on cosine similarity.
    # Logic:
    # -1.0 <= cos <  0.0 : Very low weight (0.0 - 0.1) -> Don't shift colors that are opposite
    #  0.0 <= cos <  0.5 : Low to Medium weight (0.1 - 0.5) -> Be careful with deviating hues
    #  0.5 <= cos <= 1.0 : High weight (0.5 - 1.0) -> Trust HSI when hues align
    
    # Define the mapping points (x = cos_theta, y = weight)
    # You can tune these control points
    xp = [-1.0, 0.0, 0.5, 1.0]
    yp = [ 0.0, 0.1, 0.4, 0.9]
    
    angle_weight_map = np.interp(cos_similarity, xp, yp).astype(np.float32)

    # Ensure sum_mask is the same size as the image (it might have been resized)
    if sum_mask.shape != rgb_xy.shape[:2]:
        # Resize sum_mask to match image dimensions using Nearest Neighbor to keep it boolean
        # sum_mask is boolean, so we convert to uint8 for resize, then back to bool
        sum_mask_uint8 = sum_mask.astype(np.uint8)
        sum_mask_resized = cv2.resize(sum_mask_uint8, (rgb_xy.shape[1], rgb_xy.shape[0]), interpolation=cv2.INTER_NEAREST)
        sum_mask = sum_mask_resized.astype(bool)
        
    # Note: angle_mask is removed from here because it is now applied as a continuous weight
    enhancement_mask = (dist_hsi > dist_rgb) & sum_mask & valid_hsi_xy_mask & hsi_rgb_mask
    
    # --- 3. Use the adaptive strength logic (XY Space Version) ---
    # Replaced Lab saturation with distance to white point in xy space for efficiency and consistency.
    
    # --- Constraint: Low Saturation Filter (White Point Proximity) ---
    # Filter out pixels that are too close to the white point (neutral colors)
    white_point_radius_thresh = 0.1  # Threshold radius in xy space (approx. equivalent to low saturation)
    not_neutral_mask = dist_rgb > white_point_radius_thresh
    
    # Normalize the distance to white point (proxy for saturation)
    # dist_rgb is already calculated: np.linalg.norm(rgb_xy - white_point_xy, axis=-1)
    dist_min, dist_max = np.min(dist_rgb), np.max(dist_rgb)
    
    if dist_max - dist_min < 1e-6:
        normalized_dist = np.zeros_like(dist_rgb)
    else:
        normalized_dist = (dist_rgb - dist_min) / (dist_max - dist_min)
        
    # Base adaptive strength based on saturation
    base_strength = global_strength * (1 - normalized_dist)
    
    # Apply angle-based weighting
    adaptive_strength_map = base_strength * angle_weight_map
    
    # --- 4. Apply enhancement only where the mask is True ---
    # Combine all masks:
    # - dist_hsi > dist_rgb: Target is more saturated
    # - sum_mask: Pixel is bright enough
    # - valid_hsi_xy_mask: HSI data is valid
    # - hsi_rgb_mask: Hue direction is consistent (geometric check)
    # - not_neutral_mask: Pixel is far enough from white point (not gray)
    
    final_mask = enhancement_mask & not_neutral_mask
    
    enhanced_xy = np.copy(rgb_xy)
    strength_for_enhancement = adaptive_strength_map[final_mask]
    strength_3d = strength_for_enhancement[..., np.newaxis]
    
    enhanced_xy[final_mask] = rgb_xy[final_mask] + \
                                    strength_3d * (hsi_xy_map[final_mask] - rgb_xy[final_mask])
    
    # --- 5. Reconstruct the image ---
    enhanced_xyY = np.stack([enhanced_xy[..., 0], enhanced_xy[..., 1], rgb_xyY[..., 2]], axis=-1)
    enhanced_xyz = xyY_to_xyz(enhanced_xyY)
    
    return enhanced_xyz


# 色域三角形顶点 (CIE xy)：增强时“允许的色度范围”。Rec.2020 绿色范围更大。
GAMUT_PRIMARIES = {
    "sRGB": {
        "r": np.array([0.6400, 0.3300], dtype=np.float32),
        "g": np.array([0.3000, 0.6000], dtype=np.float32),
        "b": np.array([0.1500, 0.0600], dtype=np.float32),
    },
    "Rec.2020": {
        "r": np.array([0.7080, 0.2920], dtype=np.float32),
        "g": np.array([0.1700, 0.7970], dtype=np.float32),
        "b": np.array([0.1310, 0.0460], dtype=np.float32),
    },
}


def _point_in_triangle_xy(half_points, r, g, b):
    """Barycentric: half_points (...,2) 是否在三角形 r,g,b (xy) 内。"""
    v0 = g - r
    v1 = b - r
    v2 = half_points - r
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.sum(v0 * v2, axis=-1)
    dot11 = np.dot(v1, v1)
    dot12 = np.sum(v1 * v2, axis=-1)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) & (v >= 0) & (u + v <= 1)


# --- Main Pipeline ---

def run_advanced_pipeline(
    rgb_path: Path,
    hsi_path: Path,
    output_path: Path,
    do_white_balance: bool = True,
    do_chroma_enhance: bool = True,
    enhancement_strength: float = 0.6,
    hsi_sum_thresh_factor: float = 0.1,
    output_colourspace: str = "Rec.2020",
    target_gamut: str = "Rec.2020",
):
    """
    Executes the advanced HSI->RGB correction pipeline.
    """
    print(f"\nProcessing {rgb_path.name}...")
    # 1. Load data
    hsi_cube, wavelengths = None, None  # Initialize
    try:
        rgb_img_orig = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        h, w, _ = rgb_img_orig.shape
        # rgb_img_orig = cv2.resize(rgb_img_orig, (1024, int(1024/h*w)), interpolation=cv2.INTER_LINEAR)
        if rgb_img_orig is None:
            raise IOError(f"Failed to load RGB image: {rgb_path}")
        rgb_img_orig = cv2.cvtColor(rgb_img_orig, cv2.COLOR_BGR2RGB)
        
        with hsi_path.open("rb") as f:
            hsi_data = pickle.load(f)
            
        # Data for visualization and white balance (pre-computed)
        hsi_xy_map_precomputed = np.asarray(hsi_data['cie_xy'], dtype=np.float32)
        hsi_xyz_precomputed = np.asarray(hsi_data['cie_XYZ'], dtype=np.float32)
        # hsi_rgb_data = np.asarray(hsi_data.get('rgb'), dtype=np.uint8) # No longer used
        
        # Data for new enhancement pipeline
        hsi_cube = np.asarray(hsi_data['hsi'], dtype=np.float32)
        wavelengths = np.asarray(hsi_data['chdef'], dtype=np.float32)

        # Note: We will generate hsi_rgb_data from our calculated XYZ later, 
        # instead of using the pre-computed one from the pkl file.

    except (IOError, pickle.PickleError, KeyError) as e:
        print(f"  Error loading data: {e}. Skipping.")
        return

    # 2. Pre-process and resize
    rgb_float = img_as_float32(rgb_img_orig)

    # --- 1. Normalize HSI cube channel-wise, clipping top/bottom 2% ---
    h, w, num_bands = hsi_cube.shape
    normalized_hsi_cube = np.zeros_like(hsi_cube, dtype=np.float32)
    for i in range(num_bands):
        band = hsi_cube[..., i]
        p2, p98 = np.percentile(band, [2, 98])
        if p98 <= p2:  # Handle flat or invalid channels
            normalized_hsi_cube[..., i] = 0
        else:
            band_clipped = np.clip(band, p2, p98)
            normalized_hsi_cube[..., i] = (band_clipped - p2) / (p98 - p2)

    # --- 2. Create validity mask based on sum of normalized reflectance ---
    hsi_sum = np.sum(normalized_hsi_cube, axis=-1)
    threshold = hsi_sum_thresh_factor * num_bands
    sum_mask = hsi_sum > threshold
    print(f"  {np.sum(sum_mask)} / {h*w} pixels are bright enough for enhancement.")

    # --- 3. Convert normalized HSI cube to target xy map (FAST VERSION) ---
    # Instead of using the slow colour.MultiSpectralDistributions object,
    # we implement the spectral-to-XYZ integration directly using matrix multiplication.
    
    # OPTIMIZATION: Downsample normalized HSI cube to 1/4 size (half width, half height) 
    # before heavy spectral-to-XYZ conversion to save significant computation time.
    h_small, w_small = h // 4, w // 4
    if h_small > 0 and w_small > 0:
        normalized_hsi_cube_small = cv2.resize(normalized_hsi_cube, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    else:
        normalized_hsi_cube_small = normalized_hsi_cube # Fallback for very small images
        h_small, w_small = h, w
    
    # 1. Get CMFs and Illuminant
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']

    # 2. Align CMFs and Illuminant to HSI wavelengths MANUALLY using interpolation
    # This avoids shape mismatch errors from colour.align()
    
    # Extract domain and range from CMFs and Illuminant
    cmfs_domain = cmfs.wavelengths
    cmfs_values = cmfs.values
    ill_domain = illuminant.wavelengths
    ill_values = illuminant.values

    # Create interpolation functions
    # fill_value=0.0 ensures we don't get errors if HSI range is slightly outside standard CMF range
    # We use 'cubic' interpolation because 20nm intervals are relatively wide, 
    # and CMF curves are smooth. Cubic spline fits better than linear.
    cmf_interp = interp1d(cmfs_domain, cmfs_values, axis=0, kind='cubic', bounds_error=False, fill_value=0.0)
    ill_interp = interp1d(ill_domain, ill_values, axis=0, kind='cubic', bounds_error=False, fill_value=0.0)

    # Interpolate to HSI wavelengths
    cmfs_aligned_values = cmf_interp(wavelengths)
    S = ill_interp(wavelengths)
    
    # 3. Pre-calculate the integration weights (CMF * Illuminant)
    # Shape: (num_bands, 3)
    # The standard formula includes a normalizing factor k = 100 / sum(y_bar * S)
    # XYZ = k * sum(R * S * x_bar), etc.
    
    # Ensure S is 1D array of shape (num_bands,)
    if S.ndim > 1: S = S.squeeze()
    
    x_bar, y_bar, z_bar = cmfs_aligned_values.T
    
    # Normalizing constant k
    # We assume equal wavelength spacing (d_lambda) which cancels out in the ratio, 
    # or is implicitly handled by sum.
    k = 100.0 / np.sum(y_bar * S)
    
    # Integration matrix M: (num_bands, 3)
    # Column 0: S * x_bar * k
    # Column 1: S * y_bar * k
    # Column 2: S * z_bar * k
    M = (cmfs_aligned_values * S[:, np.newaxis]) * k
    
    # 4. Calculate XYZ using Matrix Multiplication
    # hsi_flat: (num_pixels, num_bands)
    # M:        (num_bands, 3)
    # Result:   (num_pixels, 3)
    
    hsi_flat = normalized_hsi_cube_small.reshape(-1, num_bands)
    
    # This dot product replaces the entire colour.sd_to_XYZ pipeline
    xyz_from_hsi_flat = np.dot(hsi_flat, M)
    
    xyz_from_hsi = xyz_from_hsi_flat.reshape(h_small, w_small, 3)
    hsi_xy_map = xyz_to_xy(xyz_from_hsi)
    
    # Note: xyz_from_hsi and hsi_xy_map are now at 1/4 resolution.
    # They will be automatically upscaled to match the RGB image size in the next block.
    
    h, w, _ = rgb_float.shape
    if hsi_xy_map.shape[:2] != (h, w):
        hsi_xy_map = cv2.resize(hsi_xy_map, (w, h), interpolation=cv2.INTER_LINEAR)
        xyz_from_hsi = cv2.resize(xyz_from_hsi, (w, h), interpolation=cv2.INTER_LINEAR)
        # Resize sum_mask to match new dimensions
        sum_mask_uint8 = sum_mask.astype(np.uint8)
        sum_mask = cv2.resize(sum_mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
    # Generate HSI RGB data from our calculated XYZ for consistent visualization
    hsi_rgb_float = xyz_to_rgb(xyz_from_hsi)
    hsi_rgb_data = (np.clip(hsi_rgb_float, 0, 1) * 255).astype(np.uint8)

    # 3. Initial Conversion
    rgb_xyz = rgb_to_xyz(rgb_float)
    
    # --- New Step: Visualize Original Gamuts ---
    vis_output_dir = output_path.parent / "gamut_visuals"
    
    # Check if original visualizations already exist before plotting
    hsi_vis_check_file = vis_output_dir / f"{output_path.stem}_hsi_original_lab.png"
    rgb_vis_check_file = vis_output_dir / f"{output_path.stem}_rgb_original_lab.png"
    
    if hsi_vis_check_file.exists() and rgb_vis_check_file.exists():
        print("  Skipping original gamut visualization, files already exist.")
    else:
        # Plot HSI Gamut
        if hsi_rgb_data is not None:
            visualize_gamut_plots(
                xyz_data=xyz_from_hsi,
                rgb_data=hsi_rgb_data,
                plot_prefix=f"{output_path.stem}_hsi_original",
                output_dir=vis_output_dir
            )
        # Plot Original RGB Gamut
        visualize_gamut_plots(
            xyz_data=rgb_xyz,
            rgb_data=rgb_img_orig,
            plot_prefix=f"{output_path.stem}_rgb_original",
            output_dir=vis_output_dir
        )

    # --- Execute Correction Steps ---
    corrected_xyz = rgb_xyz
    
    if do_white_balance:
        # --- Using Upgraded White Balance ---
        # Use the newly calculated xyz_from_hsi which is derived from the normalized full spectrum
        corrected_xyz = correct_white_balance_gain(corrected_xyz, xyz_from_hsi)
        
        # Save intermediate white balance result
        corrected_rgb = xyz_to_rgb(corrected_xyz)
        corrected_rgb_uint8 = (np.clip(corrected_rgb, 0, 1) * 255).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Correct way to create a new path with a modified name using pathlib
        output_path_white_balance = output_path.with_name(f"{output_path.stem}_white_balance{output_path.suffix}")
        # Image.fromarray(corrected_rgb_uint8[::8,::8]).save(output_path_white_balance)
        
        # visualize_gamut_plots(
        #     xyz_data=corrected_xyz,
        #     rgb_data=corrected_rgb_uint8,
        #     plot_prefix=f"{output_path.stem}_rgb_white_balanced",
        #     output_dir=vis_output_dir
        # )
        
    if do_chroma_enhance:
        if hsi_cube is not None and wavelengths is not None:
            corrected_xyz = enhance_by_gamut_expansion(
                corrected_xyz, 
                hsi_xy_map, 
                sum_mask,
                global_strength=enhancement_strength,
                target_gamut=target_gamut,
            )
        else:
            print("  Skipping chroma enhance: HSI cube data ('hsi', 'chdef') not found.")
        
    # 4. Final Conversion and Save（Rec.2020 色域更大，绿色表现更好，适合高光谱映射结果）
    if output_colourspace == "Rec.2020":
        final_rgb = xyz_to_rec2020(corrected_xyz)
        print(f"  Output colourspace: Rec.2020 (wider gamut, better greens).")
    else:
        final_rgb = xyz_to_rgb(corrected_xyz)
    final_rgb_uint8 = (np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)
    
    # --- Optional: Visualize Corrected Gamut（与输出一致：用实际保存的 Rec.2020/sRGB 数值，不再压回 sRGB）---
    final_xyz = corrected_xyz
    visualize_gamut_plots(
        xyz_data=final_xyz,
        rgb_data=final_rgb_uint8,
        plot_prefix=f"{output_path.stem}_rgb_corrected_{output_colourspace.replace('.', '')}",
        output_dir=vis_output_dir
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_rgb_uint8[::4,::4]).save(output_path)
    print(f"  Successfully saved to {output_path}")


if __name__ == "__main__":
    from PIL import Image
    import time
    
    # --- Configuration ---
    folder_rgb = "/Volumes/PSSD/hyperspectral/chunked_exp_results/RealESRGAN_enhanced"
    folder_hsi = '/Volumes/PSSD/hyperspectral/HSI_dataset/chunked_dataset/HSI_data'
    output_dir = "/Volumes/PSSD/hyperspectral/chunked_out/RealESRGAN_HSI_Advanced"
    
    # Get list of files
    rgb_files = sorted([f for f in Path(folder_rgb).iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    start_time = time.time()
    for i, rgb_file_path in enumerate(rgb_files[1838:]):
        file_num = rgb_file_path.stem.split("_")[0]
        hsi_file_name = f"{file_num}.pkl"
        hsi_file_path = Path(folder_hsi) / hsi_file_name
        
        output_file_path = Path(output_dir) / f"{file_num}.png"
        
        if not hsi_file_path.exists():
            print(f"Skipping {file_num}, no corresponding HSI file found.")
            continue
            
        run_advanced_pipeline(
            rgb_path=rgb_file_path,
            hsi_path=hsi_file_path,
            output_path=output_file_path,
            # --- Pipeline Controls ---
            do_white_balance=True,
            do_chroma_enhance=True,
            enhancement_strength=0.9,  # GLOBAL MAX strength for EXPANSION
            output_colourspace="Rec.2020",  # 输出 Rec.2020 以更好保留绿色等宽色域
            target_gamut="Rec.2020",        # 色域约束放宽到 Rec.2020
        )
        
    end_time = time.time()
    print(f"\nPipeline finished. Processed {len(rgb_files)} images in {end_time - start_time:.2f} seconds.")
