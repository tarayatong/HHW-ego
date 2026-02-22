#!/usr/bin/env python3
"""
Convert a hyperspectral image stored in a pickle file to an RGB image.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER

    HAS_COLOUR = True
except ImportError:
    MSDS_CMFS_STANDARD_OBSERVER = None  # type: ignore
    HAS_COLOUR = False

# sRGB XYZ to RGB conversion matrix (D65, linear).
XYZ_TO_SRGB_MATRIX = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)


def load_hsi(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load hyperspectral cube and wavelength definitions from a pickle file."""
    with path.open("rb") as handle:
        data = pickle.load(handle)

    if "hsi" not in data:
        raise KeyError("Missing 'hsi' key in pickle file.")
    if "chdef" not in data:
        raise KeyError("Missing 'chdef' key in pickle file.")

    hsi = np.asarray(data["hsi"], dtype=np.float32)
    wavelengths = np.asarray(data["chdef"], dtype=np.float32)
    return hsi, wavelengths, data


def normalize_hsi_bands(hsi: np.ndarray, mode: str = "per_band") -> np.ndarray:
    """Normalize hyperspectral bands prior to colour conversion."""
    if mode == "none":
        return hsi

    if mode not in {"per_band", "global", "mean_compensation"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    hsi = np.asarray(hsi, dtype=np.float32)

    if mode == "per_band":
        reshaped = hsi.reshape(-1, hsi.shape[-1])
        min_vals = reshaped.min(axis=0)
        max_vals = reshaped.max(axis=0)
        denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
        normalized = (hsi - min_vals) / denom
    elif mode == "global":
        min_val = float(hsi.min())
        max_val = float(hsi.max())
        if max_val > min_val:
            normalized = (hsi - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(hsi, dtype=np.float32)
    else:  # mean_compensation
        band_means = hsi.reshape(-1, hsi.shape[-1]).mean(axis=0)
        overall_mean = float(np.mean(band_means))
        safe_band_means = np.where(band_means != 0.0, band_means, 1.0)
        scale = overall_mean / safe_band_means
        normalized = hsi * scale
        min_val = float(normalized.min())
        max_val = float(normalized.max())
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(normalized, dtype=np.float32)

    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def compute_delta_lambda(wavelengths: np.ndarray) -> np.ndarray:
    """Compute wavelength interval for numerical integration."""
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("Wavelength array must be one-dimensional with at least two items.")

    diffs = np.diff(wavelengths)
    last_step = diffs[-1]
    delta_lambda = np.concatenate([diffs, np.array([last_step], dtype=wavelengths.dtype)])
    return delta_lambda


def hsi_to_xyz(hsi: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Integrate spectral cube against CIE 1931 CMFs to obtain XYZ image."""
    if not HAS_COLOUR:
        raise RuntimeError("colour-science is required to integrate spectra into XYZ.")
    cmf = MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"]
    cmf_wavelengths = np.asarray(cmf.wavelengths, dtype=np.float32)
    cmf_values = np.asarray(cmf.values, dtype=np.float32)

    x_bar = np.interp(wavelengths, cmf_wavelengths, cmf_values[:, 0], left=0.0, right=0.0)
    y_bar = np.interp(wavelengths, cmf_wavelengths, cmf_values[:, 1], left=0.0, right=0.0)
    z_bar = np.interp(wavelengths, cmf_wavelengths, cmf_values[:, 2], left=0.0, right=0.0)

    delta_lambda = compute_delta_lambda(wavelengths)

    X = np.tensordot(hsi, x_bar * delta_lambda, axes=([2], [0]))
    Y = np.tensordot(hsi, y_bar * delta_lambda, axes=([2], [0]))
    Z = np.tensordot(hsi, z_bar * delta_lambda, axes=([2], [0]))

    xyz = np.stack([X, Y, Z], axis=-1)
    return xyz


def normalize_xyz(xyz: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Scale XYZ image so that the chosen percentile of Y maps to 1.0."""
    if percentile <= 0 or percentile > 100:
        raise ValueError("Percentile must be in the range (0, 100].")

    Y = xyz[..., 1]
    y_max = np.percentile(Y, percentile)
    if y_max <= 0:
        return xyz
    scale = 1.0 / y_max
    return xyz * scale


def xyz_to_srgb(xyz: np.ndarray, clip: bool = True) -> np.ndarray:
    """Transform XYZ tristimulus values to gamma-corrected sRGB."""
    linear_rgb = np.tensordot(xyz, XYZ_TO_SRGB_MATRIX, axes=([2], [1]))
    linear_rgb = np.clip(linear_rgb, 0.0, None)

    threshold = 0.0031308
    srgb = np.where(
        linear_rgb <= threshold,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    )

    if clip:
        srgb = np.clip(srgb, 0.0, 1.0)
    return srgb


def save_rgb(rgb: np.ndarray, output_path: Path) -> None:
    """Save sRGB image (float in [0,1]) to disk as PNG."""
    rgb_uint8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8).save(output_path)


def convert_hsi_to_rgb(
    input_path: Path,
    output_path: Path,
    percentile: float = 99.5,
    skip_normalization: bool = False,
    band_normalization: str = "per_band",
) -> None:
    """Full pipeline: load HSI, integrate to XYZ, normalize, convert, and save."""
    hsi, wavelengths, data = load_hsi(input_path)
    hsi = normalize_hsi_bands(hsi, mode=band_normalization)

    if HAS_COLOUR:
        xyz = hsi_to_xyz(hsi, wavelengths)
    else:
        if "cie_XYZ" not in data:
            raise RuntimeError(
                "colour-science is not available and 'cie_XYZ' is missing in the pickle file."
            )
        xyz = np.asarray(data["cie_XYZ"], dtype=np.float32)

    if not skip_normalization:
        xyz = normalize_xyz(xyz, percentile=percentile)

    srgb = xyz_to_srgb(xyz)
    save_rgb(srgb, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a hyperspectral pickle to an RGB image.")
    parser.add_argument(
        "--input",
        type=Path,
        default="HSI_dataset/HSI_data/169.pkl",
        help="Path to the input pickle file (e.g. HSI_dataset/HSI_data/0.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="color_analysis/HSI_169.png",
        help="Path to the output RGB image (PNG).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile of Y used for normalization (default: 99.5).",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Disable dynamic range normalization before RGB conversion.",
    )
    parser.add_argument(
        "--band-normalization",
        choices=["per_band", "global", "mean_compensation", "none"],
        default="per_band",
        help="How to normalize spectral bands before conversion (default: per_band).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_hsi_to_rgb(
        input_path=args.input,
        output_path=args.output,
        percentile=args.percentile,
        skip_normalization=args.skip_normalization,
        band_normalization=args.band_normalization,
    )
    print(f"Saved RGB image to {args.output}")


if __name__ == "__main__":
    main()

