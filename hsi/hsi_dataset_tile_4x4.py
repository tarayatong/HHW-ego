"""
按「要处理的目录 / 保存目录」列表对图像做 4x4 切分，输出瓦片顺延编号 0 ~ N*16-1。

- 处理逻辑：列出 (输入目录, 输出目录, 扩展名) 任务列表，逐项处理；当前仅包含新的实验结果目录（不依赖原数据集根目录）。
- 编号规则：每目录内按「文字_编号」排序（如 enhanced_0, enhanced_1 或 0, 1），第 i 个文件 → 瓦片 i*16..i*16+15。
- 已处理过的输出目录（存在 .tile_4x4_done）会跳过。
"""

import os
import re
import pickle
from pathlib import Path

import numpy as np

# 尝试用 cv2 读图，若无则用 PIL
try:
    import cv2
    def read_image(path):
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Failed to load {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def write_image(path, img):
        if img.ndim == 3 and img.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        cv2.imwrite(str(path), img_bgr)
except ImportError:
    from PIL import Image
    def read_image(path):
        img = Image.open(path)
        return np.array(img)
    def write_image(path, img):
        Image.fromarray(np.asarray(img)).save(path)


# 默认 HSI_dataset 路径（输入数据根目录）
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/Volumes/PSSD/hyperspectral/HSI_dataset")
if not DEFAULT_DATASET_ROOT.exists():
    DEFAULT_DATASET_ROOT = SCRIPT_DIR.parent / "HSI_dataset"

# 切分结果输出根目录（不再保存在数据集根目录下的 *_4x4 文件夹）
DEFAULT_OUTPUT_ROOT = Path("/Volumes/PSSD/hyperspectral/HSI_dataset/chunked_dataset")

# RealESRGAN 等实验结果的切分：输入目录与输出目录
DEFAULT_EXP_INPUT = Path("/Volumes/PSSD/hyperspectral/exp_results/RealESRGAN_enhanced")
DEFAULT_EXP_OUTPUT = Path("/Volumes/PSSD/hyperspectral/chunked_exp_results/RealESRGAN_enhanced")

SUBFOLDERS = [
    "HSI_data",
    "HDR_img",
    "EmdoorVR_img",
    "RayNeo_img",
    "Glasses_img",
]

# 每个子文件夹对应的扩展名
EXT_BY_FOLDER = {
    "HSI_data": ".pkl",
    "HDR_img": ".jpg",
    "EmdoorVR_img": ".jpg",
    "RayNeo_img": ".jpg",
    "Glasses_img": ".jpg",
}


def numeric_sort_key(name_without_ext):
    """提取数字用于排序；纯数字按数值，否则按字符串，保证类型可比较。"""
    m = re.match(r"^(\d+)$", name_without_ext)
    if m:
        return (0, int(m.group(1)))
    return (1, name_without_ext)


def stem_numeric_sort_key(name_without_ext: str):
    """
    支持「编号_文字」「文字_编号」的排序键：纯数字、开头数字_、末尾_数字、末尾数字 均按数值排序。
    例如：0, 0_out, 100_out, 101_out, enhanced_1 -> 按 0,0,100,101,1 排序。
    """
    s = name_without_ext.strip()
    if re.match(r"^\d+$", s):
        return (0, int(s))
    m = re.match(r"^(\d+)_", s)
    if m:
        return (0, int(m.group(1)))
    m = re.search(r"_(\d+)$", s)
    if m:
        return (0, int(m.group(1)))
    m = re.search(r"(\d+)$", s)
    if m:
        return (0, int(m.group(1)))
    return (1, s)


def stem_to_index(stem: str) -> int:
    """
    从 stem 提取编号用于瓦片下标。支持：
    - 纯数字：0, 1
    - 编号_文字：0_out, 100_out, 101_out → 取开头数字
    - 文字_编号：enhanced_0, frame_100 → 取末尾数字
    """
    s = stem.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    m = re.match(r"^(\d+)_", s)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)$", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    return 0


def get_sorted_indices(folder_path, ext, sort_key=None):
    """返回按数字升序排列的文件名（不含扩展名）列表。sort_key 默认用 stem_numeric_sort_key（支持文字_编号）。"""
    if not folder_path.exists():
        return []
    if sort_key is None:
        sort_key = stem_numeric_sort_key
    names = []
    for f in folder_path.iterdir():
        if f.suffix.lower() == ext.lower() and f.is_file():
            names.append(f.stem)
    names.sort(key=sort_key)
    return names


def format_size(n_bytes: int) -> str:
    """将字节数格式化为可读字符串。"""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.2f} KB"
    return f"{n_bytes / (1024 * 1024):.2f} MB"


# 与 hsi 同尺寸切分时的目标瓦片尺寸（上下各拓宽 2 像素后 296+4=300，宽 400）
TILE_H, TILE_W = 300, 400
OVERLAP_UP_DOWN = 2


def split_4x4_grid(arr):
    """
    将二维或三维数组按 4x4 切分为 16 块（无重叠）。
    arr: (H, W) 或 (H, W, C)。返回 list of 16 个数组。保持原 dtype。
    用于与 hsi 尺寸不同的空间数组（如 demosaic_gray）。
    """
    h, w = arr.shape[0], arr.shape[1]
    if h < 4 or w < 4:
        return None
    h4, w4 = h // 4, w // 4
    tiles = []
    for i in range(4):
        for j in range(4):
            r0, r1 = i * h4, (i + 1) * h4
            c0, c1 = j * w4, (j + 1) * w4
            tiles.append(arr[r0:r1, c0:c1].copy())
    return tiles


def split_4x4_grid_with_overlap(arr, overlap=OVERLAP_UP_DOWN, target_h=TILE_H, target_w=TILE_W):
    """
    将二维或三维数组按 4x4 切分，每个区域上下各拓宽 overlap 像素，输出统一为 target_h x target_w。
    用于与 hsi 同尺寸的数组（1184x1600 → 16 块 300x400，相邻块在竖直方向重叠 4 像素）。
    边界不足时用边缘像素填充到 target_h x target_w。
    """
    h, w = arr.shape[0], arr.shape[1]
    if h < 4 or w < 4:
        return None
    h4, w4 = h // 4, w // 4
    # 每块名义高度 h4，上下各扩 overlap，即 h4 + 2*overlap = 300（当 h4=296, overlap=2）
    block_h = h4 + 2 * overlap
    tiles = []
    for i in range(4):
        for j in range(4):
            r0 = max(0, i * h4 - overlap)
            r1 = min(h, r0 + block_h)
            c0 = j * w4
            c1 = min(w, (j + 1) * w4)
            patch = arr[r0:r1, c0:c1].copy()
            need_h = target_h - patch.shape[0]
            need_w = target_w - patch.shape[1]
            if need_h > 0 or need_w > 0:
                pad_width = [(0, 0)] * arr.ndim
                pad_width[0] = (0, need_h)
                pad_width[1] = (0, need_w)
                patch = np.pad(patch, pad_width, mode="edge")
            elif need_h < 0 or need_w < 0:
                patch = patch[:target_h, :target_w]
            tiles.append(patch)
    return tiles


def _is_spatial_array(arr, ref_h=None, ref_w=None):
    """判断是否为空间数组（2D 或 3D），可选与参考尺寸一致。"""
    if not isinstance(arr, np.ndarray) or arr.ndim not in (2, 3):
        return False
    if ref_h is not None and ref_w is not None:
        return arr.shape[0] == ref_h and arr.shape[1] == ref_w
    return True


def tile_pkl_folder(src_dir: Path, out_dir: Path, start_index: int = 0):
    """
    将 HSI_data 下的 .pkl 按 4x4 切分并顺延编号保存。
    编号按源文件顺序：第 i 个 pkl 对应瓦片 i*16 .. i*16+15，与其它文件夹对齐。
    start_index > 0 时从第 start_index 个 pkl 开始处理（用于断点续跑）。
    自动识别 pkl 内所有空间数组：与 hsi 同尺寸的用同一网格切分，尺寸不同的按自身 4x4 切分；
    非空间键（pkg_type, version, mode, chdef 等）原样复制。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names_full = get_sorted_indices(src_dir, ".pkl")
    if not names_full:
        print(f"  未找到 .pkl 文件: {src_dir}")
        return

    test_one = getattr(tile_pkl_folder, "_test_one", False)
    if test_one:
        names = names_full[:1]
        print(f"  [测试] 仅处理第 1 个 pkl: {names[0]}.pkl")
        start_index = 0
    else:
        names = names_full[start_index:]
        if start_index > 0:
            print(f"  [续跑] 从源图索引 {start_index} 开始（共 {len(names)} 个 pkl，瓦片编号 {start_index*16} 起）")
    total_tiles = 0
    for i, stem in enumerate(names):
        global_i = start_index + i
        pkl_path = src_dir / f"{stem}.pkl"
        if not pkl_path.is_file():
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print("processing", pkl_path.name)
        if "hsi" not in data:
            print(f"  跳过 {pkl_path.name}：缺少 'hsi' 键")
            continue

        hsi = np.asarray(data["hsi"])
        h, w = hsi.shape[0], hsi.shape[1]
        if h < 4 or w < 4:
            print(f"  跳过 {pkl_path.name}：尺寸 {h}x{w} 不足以 4x4 切分")
            continue

        # 与 hsi 同尺寸的空间键 → 用同一 4x4 网格
        same_grid_keys = [
            k for k, v in data.items()
            if _is_spatial_array(v, h, w)
        ]
        # 其他空间数组（尺寸与 hsi 不同）→ 按自身尺寸 4x4
        other_spatial_keys = [
            k for k, v in data.items()
            if _is_spatial_array(v) and k not in same_grid_keys
        ]
        # 非空间键
        non_spatial_keys = [
            k for k in data.keys()
            if k not in same_grid_keys and k not in other_spatial_keys
        ]

        tile_data_list = [dict() for _ in range(16)]

        # 1) 同尺寸键：同一网格切分，上下各拓宽 2 像素，输出 300x400
        for key in same_grid_keys:
            arr = np.asarray(data[key])
            tiles = split_4x4_grid_with_overlap(arr, overlap=OVERLAP_UP_DOWN, target_h=TILE_H, target_w=TILE_W)
            if tiles is None:
                continue
            for t_idx, tile in enumerate(tiles):
                tile_data_list[t_idx][key] = tile

        # 2) 不同尺寸的空间键：按自身 4x4 切分
        for key in other_spatial_keys:
            arr = np.asarray(data[key])
            tiles = split_4x4_grid(arr)
            if tiles is None:
                print(f"  警告：{pkl_path.name} 中 {key} 尺寸 {arr.shape} 不足以 4x4，已跳过")
                continue
            for t_idx, tile in enumerate(tiles):
                tile_data_list[t_idx][key] = tile

        # 3) 非空间键原样复制
        for key in non_spatial_keys:
            val = data[key]
            for t_dict in tile_data_list:
                # 不可变类型直接赋，数组等复制一份避免引用同一对象
                if isinstance(val, np.ndarray):
                    t_dict[key] = val.copy()
                else:
                    t_dict[key] = val

        # 测试模式：打印第一个 tile 的各键 shape
        if test_one:
            t0 = tile_data_list[0]
            print("\n  Tile 0 各键 shape / 类型:")
            for k in sorted(t0.keys()):
                v = t0[k]
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"    {k}: type={type(v).__name__}, value={repr(v)[:50]}")
            print()

        global_idx_start = global_i * 16
        for t_idx, t_dict in enumerate(tile_data_list):
            out_path = out_dir / f"{global_idx_start + t_idx}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(t_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_tiles += 16

        if test_one:
            out_size = sum(os.path.getsize(out_dir / f"{global_idx_start + t_idx}.pkl") for t_idx in range(16))
            print(f"  处理后: 16 个瓦片, 单瓦片尺寸 {TILE_H}×{TILE_W} (hsi/rgb 等), 本目录输出总大小 {format_size(out_size)}")

    print(f"  {src_dir.name}: {len(names)} 个 pkl -> {total_tiles} 个瓦片 -> {out_dir}")
    if total_tiles > 0 and not test_one:
        _mark_folder_done(out_dir)


def tile_image_folder(
    src_dir: Path, out_dir: Path, ext: str,
    canonical_stems: list,
    test_one: bool = False,
    start_index: int = 0,
):
    """
    将图像文件夹按 4x4 切分并顺延编号保存。

    处理逻辑：
    - 规范顺序 canonical_stems 来自 HSI_data 的 pkl 列表（如 "0","1",...,"N-1"）。
    - 按该顺序遍历每个 stem，若本目录存在 {stem}{ext}（如 0.jpg, 1.jpg）则处理并写入瓦片 i*16..i*16+15。
    - start_index > 0 时只处理 canonical_stems[start_index:] 且编号从 start_index*16 起（用于断点续跑）。
    - test_one 为 True 时仅处理「规范顺序中第一个存在的」一张图。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not canonical_stems:
        print(f"  无规范顺序，跳过: {src_dir}")
        return

    if test_one:
        # 只处理第一个在本地存在的 stem，且后续只遍历这一个（避免改到调用方传入的列表）
        stems_to_process = None
        canonical_stems_start_idx = 0
        for i, stem in enumerate(canonical_stems):
            if (src_dir / f"{stem}{ext}").is_file():
                stems_to_process = [stem]
                canonical_stems_start_idx = i
                break
        if not stems_to_process:
            print(f"  未找到 {ext} 文件: {src_dir}")
            return
        print(f"  [测试] 仅处理第 1 张图: {stems_to_process[0]}{ext}")
    else:
        stems_to_process = canonical_stems[start_index:]  # 从 start_index 起，与 HSI_data 续跑对齐
        canonical_stems_start_idx = start_index
        if start_index > 0:
            print(f"  [续跑] 从源图索引 {start_index} 开始（瓦片编号 {start_index*16} 起）")

    total_tiles = 0
    processed_count = 0
    for idx_in_canonical, stem in enumerate(stems_to_process):
        img_path = src_dir / f"{stem}{ext}"
        if not img_path.is_file():
            continue
        i = canonical_stems_start_idx + idx_in_canonical
        processed_count += 1
        print(f"    处理第 {processed_count} 张: {stem}{ext} -> 瓦片 {i*16}..{i*16+15}")
        try:
            img = read_image(img_path)
        except Exception as e:
            print(f"  读取失败 {img_path}: {e}")
            continue

        h, w = img.shape[0], img.shape[1]
        if h < 4 or w < 4:
            print(f"  跳过 {img_path.name}：尺寸 {h}x{w} 不足以 4x4 切分")
            continue

        tiles = split_4x4_grid(img)
        th, tw = tiles[0].shape[0], tiles[0].shape[1]
        global_idx_start = i * 16
        for t_idx, tile in enumerate(tiles):
            out_path = out_dir / f"{global_idx_start + t_idx}{ext}"
            write_image(out_path, tile)
        total_tiles += 16

        if test_one:
            out_size = sum(os.path.getsize(out_dir / f"{global_idx_start + t_idx}{ext}") for t_idx in range(16))
            print(f"  处理后: 16 张瓦片, 单瓦片尺寸 {th}×{tw}, 本目录输出总大小 {format_size(out_size)}")

    num_sources = total_tiles // 16
    print(f"  {src_dir.name}: {num_sources} 张图 -> {total_tiles} 张瓦片（与规范顺序对齐）-> {out_dir}")
    if total_tiles > 0 and not test_one:
        _mark_folder_done(out_dir)


# Sentinel：存在则视为该输出目录已处理完毕，下次跳过
TILE_DONE_SENTINEL = ".tile_4x4_done"


def _mark_folder_done(out_dir: Path):
    """写入完成标记，用 write_text 避免部分文件系统上 touch 无效。"""
    try:
        (out_dir / TILE_DONE_SENTINEL).write_text("", encoding="utf-8")
    except Exception:
        try:
            (out_dir / TILE_DONE_SENTINEL).touch()
        except Exception:
            pass


def tile_image_folder_by_position(
    src_dir: Path, out_dir: Path, ext: str,
    test_one: bool = False,
):
    """
    按本目录内文件自身排序顺序做 4x4 切分并顺延编号保存。
    编号规则：从 stem 提取数字（文字_编号 或末尾数字），该数字 n 对应瓦片 n*16 .. n*16+15。
    例如 0_out -> 0..15，100_out -> 1600..1615，101_out -> 1616..1631。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # 扩展名规范化（保证带点）
    ext = ext if ext.startswith(".") else f".{ext}"
    names = get_sorted_indices(src_dir, ext)
    if not names and ext.lower() == ".png":
        names = get_sorted_indices(src_dir, ".jpg")
        if names:
            ext = ".jpg"
            print(f"  未找到 .png，改用 .jpg: {src_dir}")
    if not names and ext.lower() == ".jpg":
        names = get_sorted_indices(src_dir, ".png")
        if names:
            ext = ".png"
            print(f"  未找到 .jpg，改用 .png: {src_dir}")
    if not names:
        print(f"  未找到 {ext} 文件: {src_dir}")
        return

    if test_one:
        names = names[:1]
        print(f"  [测试] 仅处理第 1 张: {names[0]}{ext}")
    total_tiles = 0
    for stem in names:
        img_path = src_dir / f"{stem}{ext}"
        if not img_path.is_file():
            continue
        num = stem_to_index(stem)
        global_idx_start = num * 16
        try:
            img = read_image(img_path)
        except Exception as e:
            print(f"  读取失败 {img_path}: {e}")
            continue
        h, w = img.shape[0], img.shape[1]
        if h < 4 or w < 4:
            print(f"  跳过 {img_path.name}：尺寸 {h}x{w} 不足以 4x4 切分")
            continue
        tiles = split_4x4_grid(img)
        for t_idx, tile in enumerate(tiles):
            out_path = out_dir / f"{global_idx_start + t_idx}{ext}"
            write_image(out_path, tile)
        total_tiles += 16
        if not test_one and total_tiles // 16 <= 3:
            print(f"    处理: {stem}{ext} -> 瓦片 {global_idx_start}..{global_idx_start+15}")
    num_sources = total_tiles // 16
    print(f"  {src_dir.name}: {num_sources} 张图 -> {total_tiles} 张瓦片（按本目录排序）-> {out_dir}")
    if total_tiles > 0 and not test_one:
        _mark_folder_done(out_dir)


# 要处理的目录列表：(输入目录, 输出目录, 扩展名)，与根目录无关，可只含新的实验结果目录
def _build_process_tasks():
    """构建 (input_dir, output_dir, ext) 列表，当前仅包含实验结果目录。"""
    exp_input = getattr(main, "_exp_input", None)
    exp_output = getattr(main, "_exp_output", None)
    exp_ext = getattr(main, "_exp_ext", ".png")
    if exp_input is None or exp_output is None:
        return []
    return [(Path(exp_input), Path(exp_output), exp_ext)]


def main():
    """
    按「要处理的目录 / 保存目录」列表逐项处理：每项 4x4 切分，编号按本目录内「文字_编号」排序。
    当前列表仅包含新的实验结果目录（不依赖原 HSI_dataset 根目录）。
    """
    tasks = _build_process_tasks()
    if not tasks:
        print("未配置要处理的目录（需设置 --exp-input 与 --exp-output，或勿使用 --no-exp）。")
        return

    test_one = getattr(main, "_test_one", False)
    print("4x4 切分：按「要处理的目录」列表逐项处理，编号规则为「文字_编号」排序后第 i 个 → 瓦片 i*16..i*16+15\n")
    if test_one:
        print("【测试模式】每项仅处理第 1 张图\n")

    for src_dir, out_dir, ext in tasks:
        label = src_dir.name or str(src_dir)
        if not src_dir.exists():
            print(f"  跳过（目录不存在）: {src_dir}")
            continue
        if (out_dir / TILE_DONE_SENTINEL).exists():
            print(f"  {label}: 已处理过，跳过 -> {out_dir}")
            continue
        print(f"处理: {src_dir} -> {out_dir}（扩展名 {ext}）")
        tile_image_folder_by_position(src_dir, out_dir, ext, test_one=test_one)

    print("\n全部完成。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="按「要处理的目录」列表做 4x4 切分，编号支持文字_编号；当前仅包含实验结果目录。"
    )
    parser.add_argument(
        "--exp-input",
        type=Path,
        default=DEFAULT_EXP_INPUT,
        help=f"要处理的图像目录（输入），默认: {DEFAULT_EXP_INPUT}",
    )
    parser.add_argument(
        "--exp-output",
        type=Path,
        default=DEFAULT_EXP_OUTPUT,
        help=f"切分结果保存目录（输出），默认: {DEFAULT_EXP_OUTPUT}",
    )
    parser.add_argument(
        "--exp-ext",
        type=str,
        default=".png",
        help="输入图像扩展名（支持文字_编号文件名），默认: .png",
    )
    parser.add_argument(
        "--test-one",
        action="store_true",
        help="仅处理每个目录的第 1 张图（快速验证）",
    )
    parser.add_argument(
        "--no-exp",
        action="store_true",
        help="不处理任何目录（清空任务列表，仅打印未配置）",
    )
    args = parser.parse_args()
    main._test_one = args.test_one
    main._exp_input = None if args.no_exp else args.exp_input
    main._exp_output = None if args.no_exp else args.exp_output
    main._exp_ext = args.exp_ext
    main()
