"""
测试脚本：每个目录下取一张图/一个 pkl，按 4x4 切分，打印每块输出尺寸并保存到对应 _4x4 目录下的 test_one/。

用法:
  python test_tile_sizes.py [--root /path/to/HSI_dataset]
"""

import pickle
from pathlib import Path

import numpy as np

from hsi_dataset_tile_4x4 import (
    DEFAULT_DATASET_ROOT,
    EXT_BY_FOLDER,
    OVERLAP_UP_DOWN,
    SUBFOLDERS,
    TILE_H,
    TILE_W,
    _is_spatial_array,
    get_sorted_indices,
    read_image,
    split_4x4_grid,
    split_4x4_grid_with_overlap,
    write_image,
)


def test_one_pkl(src_dir: Path, out_dir: Path) -> list[str]:
    """
    对第一个 pkl 做 4x4 切分，保存到 out_dir，返回每块尺寸说明列表。
    """
    names = get_sorted_indices(src_dir, ".pkl")
    if not names:
        return [f"  未找到 .pkl: {src_dir}"]
    stem = names[0]
    pkl_path = src_dir / f"{stem}.pkl"
    if not pkl_path.is_file():
        return [f"  文件不存在: {pkl_path}"]

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "hsi" not in data:
        return [f"  缺少 'hsi' 键: {pkl_path.name}"]

    hsi = np.asarray(data["hsi"])
    h, w = hsi.shape[0], hsi.shape[1]
    if h < 4 or w < 4:
        return [f"  尺寸不足 4x4: {h}x{w}"]

    same_grid_keys = [k for k, v in data.items() if _is_spatial_array(v, h, w)]
    other_spatial_keys = [
        k for k, v in data.items()
        if _is_spatial_array(v) and k not in same_grid_keys
    ]
    non_spatial_keys = [
        k for k in data.keys()
        if k not in same_grid_keys and k not in other_spatial_keys
    ]

    tile_data_list = [dict() for _ in range(16)]

    for key in same_grid_keys:
        arr = np.asarray(data[key])
        tiles = split_4x4_grid_with_overlap(
            arr, overlap=OVERLAP_UP_DOWN, target_h=TILE_H, target_w=TILE_W
        )
        if tiles is None:
            continue
        for t_idx, tile in enumerate(tiles):
            tile_data_list[t_idx][key] = tile

    for key in other_spatial_keys:
        arr = np.asarray(data[key])
        tiles = split_4x4_grid(arr)
        if tiles is None:
            continue
        for t_idx, tile in enumerate(tiles):
            tile_data_list[t_idx][key] = tile

    for key in non_spatial_keys:
        val = data[key]
        for t_dict in tile_data_list:
            t_dict[key] = val.copy() if isinstance(val, np.ndarray) else val

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    t0 = tile_data_list[0]
    lines.append(f"  输入: {pkl_path.name} ({h}x{w}x{hsi.shape[2]})")
    lines.append("  4x4 切分后 Tile 0 各键尺寸:")
    for k in sorted(t0.keys()):
        v = t0[k]
        if hasattr(v, "shape"):
            lines.append(f"    {k}: {v.shape} dtype={getattr(v, 'dtype', None)}")
        else:
            lines.append(f"    {k}: (非数组)")
    for idx, t_dict in enumerate(tile_data_list):
        out_path = out_dir / f"{idx}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(t_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    lines.append(f"  已保存 16 个 pkl -> {out_dir}")
    return lines


def test_one_image_folder(src_dir: Path, out_dir: Path, ext: str) -> list[str]:
    """
    对第一张图做 4x4 切分，保存到 out_dir，返回每块尺寸说明列表。
    """
    names = get_sorted_indices(src_dir, ext)
    if not names:
        return [f"  未找到 {ext}: {src_dir}"]
    stem = names[0]
    img_path = src_dir / f"{stem}{ext}"
    if not img_path.is_file():
        return [f"  文件不存在: {img_path}"]

    try:
        img = read_image(img_path)
    except Exception as e:
        return [f"  读取失败: {e}"]

    h, w = img.shape[0], img.shape[1]
    if h < 4 or w < 4:
        return [f"  尺寸不足 4x4: {h}x{w}"]

    tiles = split_4x4_grid(img)
    if not tiles:
        return [f"  切分失败"]

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"  输入: {img_path.name} ({h}x{w}x{img.shape[2] if img.ndim == 3 else 1})")
    lines.append("  4x4 切分后每块尺寸:")
    for idx, tile in enumerate(tiles):
        lines.append(f"    tile[{idx}]: {tile.shape}")
        out_path = out_dir / f"{idx}{ext}"
        write_image(out_path, tile)
    lines.append(f"  已保存 16 张图 -> {out_dir}")
    return lines


def main(dataset_root: Path = None):
    if dataset_root is None:
        dataset_root = DEFAULT_DATASET_ROOT
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        print(f"数据集根目录不存在: {dataset_root}")
        return

    print(f"数据集根目录: {dataset_root}")
    print("每个目录取 1 张图/1 个 pkl，4x4 切分并保存到对应 _4x4/test_one/\n")

    report_lines = []

    for sub in SUBFOLDERS:
        src_dir = dataset_root / sub
        # 保存到对应 _4x4 目录下的 test_one 子目录
        out_dir = dataset_root / f"{sub}_4x4" / "test_one"
        ext = EXT_BY_FOLDER.get(sub, ".jpg")

        print(f"--- {sub} ---")
        if not src_dir.exists():
            print(f"  跳过（目录不存在）: {src_dir}\n")
            report_lines.append(f"{sub}: 目录不存在")
            continue

        if sub == "HSI_data":
            lines = test_one_pkl(src_dir, out_dir)
        else:
            lines = test_one_image_folder(src_dir, out_dir, ext)

        for line in lines:
            print(line)
            report_lines.append(f"{sub} | {line.strip()}")
        print()

    # 保存汇总到根目录下一份 sizes_report.txt
    report_path = dataset_root / "test_one_sizes_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"尺寸汇总已保存: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试各目录 4x4 切分输出尺寸并保存到对应目录")
    parser.add_argument("--root", type=Path, default=None, help="HSI_dataset 根目录")
    args = parser.parse_args()
    main(dataset_root=args.root)
