#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WorldStrat HR/LR 페어 생성기
- CSV: version2_cleaned.csv (필수 컬럼: tile, split)
- HR:  HR_BASE_DIR / <tile> / <tile>_rgb.png  (fallback: HR_BASE_DIR / <tile>_rgb.png, 또는 <tile>.tif)
- LR:  LR_BASE_DIR / <tile> / L2A / <tile>-<sub>-L2A_data.tiff
      -> -CLM.tiff 기반으로 '구름 비율이 0'인 모든 revisit(sub)을 선택
- 저장: <OUTPUT_DIR>/<split>/{GT,LR}/<tile>-<sub>.png  (GT와 LR 동일 파일명)
- 크기: HR 600×600, LR 150×150 resize
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import tifffile as tiff
import imageio.v2 as imageio
import cv2
from tqdm.auto import tqdm  # 주피터/터미널 자동 감지

# =========================
# 기본 경로 (요청 사양)
# =========================
HR_BASE_DIR = Path("C:/AIFFEL/worldstrat_kaggle/versions/1/hr_dataset/12bit")
LR_BASE_DIR = Path("C:/AIFFEL/worldstrat_kaggle/versions/1/lr_dataset")
TILE_INFO_CSV = Path("C:/AIFFEL/worldstrat_kaggle/versions/version2_cleaned.csv")

HR_SUFFIX = "_rgb.png"
LR_SUFFIX = "-L2A_data.tiff"
CLM_SUFFIX = "-CLM.tiff"

# 타깃 크기
HR_TARGET = 600
LR_TARGET = 150

# =========================
# IO helpers
# =========================
def read_tiff(path: Path) -> Optional[np.ndarray]:
    try:
        return tiff.imread(str(path))
    except Exception as e:
        tqdm.write(f"[WARN] TIFF read fail: {path} ({e})")
        return None

def read_img_any(path: Path) -> Optional[np.ndarray]:
    try:
        return imageio.imread(path)
    except Exception as e:
        tqdm.write(f"[WARN] Image read fail: {path} ({e})")
        return None

def drop_alpha_if_any(arr: np.ndarray) -> np.ndarray:
    # RGBA -> RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., :3]
    return arr

def ensure_uint8_rgb(arr: np.ndarray) -> Optional[np.ndarray]:
    """RGB 8-bit로 강제 변환. 흑백이면 3채널 복제, RGBA면 알파 제거."""
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = drop_alpha_if_any(arr)
    if arr.dtype == np.uint8:
        return arr
    # 채널별 min-max로 8비트 변환
    arr = arr.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        ch = arr[..., c]
        mn, mx = float(np.min(ch)), float(np.max(ch))
        if mx > mn:
            out[..., c] = (ch - mn) / (mx - mn)
        else:
            out[..., c] = 0.0
    return (out * 255.0 + 0.5).astype(np.uint8)

def normalize_minmax_per_channel(img: np.ndarray) -> np.ndarray:
    """[H,W,C] -> 채널별 min-max 정규화 후 [0,1] float32"""
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[..., c]
        mn, mx = float(ch.min()), float(ch.max())
        out[..., c] = (ch - mn) / (mx - mn) if mx > mn else 0.0
    return out

def to_uint8_from01(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
    return (img01 * 255.0 + 0.5).astype(np.uint8)

# =========================
# Path helpers
# =========================
def hr_path_for_tile(tile: str, hr_base: Path) -> Optional[Path]:
    # 우선순위: HR_BASE/tile/tile_rgb.png -> HR_BASE/tile_rgb.png -> HR_BASE/tile/tile.tif -> HR_BASE/tile.tif
    p1 = hr_base / tile / f"{tile}{HR_SUFFIX}"
    if p1.exists():
        return p1
    p2 = hr_base / f"{tile}{HR_SUFFIX}"
    if p2.exists():
        return p2
    p3 = hr_base / tile / f"{tile}.tif"
    if p3.exists():
        return p3
    p4 = hr_base / f"{tile}.tif"
    if p4.exists():
        return p4
    tqdm.write(f"[MISS][HR] {tile}")
    return None

def list_lr_revisits(tile: str, lr_base: Path) -> List[Tuple[int, Path, Path]]:
    l2a_dir = lr_base / tile / "L2A"
    if not l2a_dir.exists():
        return []
    out = []
    for p in sorted(l2a_dir.glob(f"{tile}-*-L2A_data.tiff")):
        m = re.match(rf"^{re.escape(tile)}-(\d+)-L2A_data\.tiff$", p.name)
        if not m:
            continue
        sub = int(m.group(1))
        clm = l2a_dir / f"{tile}-{sub}{CLM_SUFFIX}"
        out.append((sub, p, clm))
    return out

def cloud_fraction(clm_path: Path) -> float:
    if not clm_path.exists():
        return float("inf")
    clm = read_tiff(clm_path)
    if clm is None:
        return float("inf")
    clouds = (clm > 0).astype(np.float32)
    return float(clouds.mean())

def select_zero_cloud_revisits(tile: str, lr_base: Path) -> List[Tuple[int, Path, Path, float]]:
    """구름 비율이 0.0인 revisit만 반환: (sub, l2a_path, clm_path, frac)"""
    cands = list_lr_revisits(tile, lr_base)
    zeros = []
    for sub, l2a, clm in cands:
        frac = cloud_fraction(clm)
        # float 안전 여유
        if np.isfinite(frac) and frac <= 0.0 + 1e-12:
            zeros.append((sub, l2a, clm, frac))
    # 정렬은 sub 기준
    zeros.sort(key=lambda x: x[0])
    return zeros

# =========================
# Resize helper
# =========================
def resize_square(img: np.ndarray, target: int) -> np.ndarray:
    """정확히 target×target으로 리사이즈. 다운스케일은 AREA, 업스케일은 CUBIC."""
    h, w = img.shape[:2]
    if h > target or w > target:
        interp = cv2.INTER_AREA
    elif h < target or w < target:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_NEAREST
    # OpenCV는 (width, height)
    return cv2.resize(img, (target, target), interpolation=interp)

# =========================
# LR 변환
# =========================
def lr_rgb_from_l2a(l2a_path: Path, band_order: Tuple[int, int, int]) -> Optional[np.ndarray]:
    arr = read_tiff(l2a_path)
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        tqdm.write(f"[WARN] Unexpected LR shape {arr.shape} at {l2a_path}")
        return None
    try:
        r = arr[..., band_order[0]]
        g = arr[..., band_order[1]]
        b = arr[..., band_order[2]]
    except Exception as e:
        tqdm.write(f"[WARN] Bad band_order {band_order} for shape {arr.shape}: {e}")
        return None
    rgb01 = normalize_minmax_per_channel(np.stack([r, g, b], axis=-1))
    return to_uint8_from01(rgb01)

# =========================
# Core
# =========================
def build_pairs_zero_cloud_all_revisits(
    csv_path: Path,
    hr_base: Path,
    lr_base: Path,
    output_dir: Path,
    band_order: Tuple[int, int, int] = (3, 2, 1),
    hr_target: int = HR_TARGET,
    lr_target: int = LR_TARGET,
) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    req = {"tile", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns {req}, got {list(df.columns)}")

    counts = {"train": 0, "val": 0, "test": 0, "other": 0}
    tiles_with_zero = 0
    pbar = tqdm(total=len(df), desc="Building pairs (cloud==0, all revisits)", unit="tile")

    for _, row in df.iterrows():
        tile = str(row["tile"])
        split = str(row["split"]).strip().lower()
        split_dir = split if split in ("train", "val", "test") else "other"

        gt_out_dir = output_dir / split_dir / "GT"
        lr_out_dir = output_dir / split_dir / "LR"
        gt_out_dir.mkdir(parents=True, exist_ok=True)
        lr_out_dir.mkdir(parents=True, exist_ok=True)

        # ---- HR ----
        hrp = hr_path_for_tile(tile, hr_base)
        if hrp is None:
            pbar.update(1)
            pbar.set_postfix_str(f"tile={tile} split={split_dir} HR=miss")
            continue
        hr_img_raw = read_img_any(hrp) if hrp.suffix.lower() != ".tif" else read_tiff(hrp)
        if hr_img_raw is None:
            pbar.update(1)
            pbar.set_postfix_str(f"tile={tile} split={split_dir} HR=read_fail")
            continue
        hr_img_uint8 = ensure_uint8_rgb(hr_img_raw)
        if hr_img_uint8 is None:
            pbar.update(1)
            pbar.set_postfix_str(f"tile={tile} split={split_dir} HR=convert_fail")
            continue
        hr_img_resized = resize_square(hr_img_uint8, hr_target)

        # ---- LR: 구름 0인 모든 revisit 찾기 ----
        zeros = select_zero_cloud_revisits(tile, lr_base)
        saved_here = 0
        for subnum, l2a_path, clm_path, cloud_frac in zeros:
            lr_img = lr_rgb_from_l2a(l2a_path, band_order=band_order)
            if lr_img is None:
                tqdm.write(f"[LR-READ-FAIL] {tile}-{subnum} @ {l2a_path}")
                continue
            lr_img_resized = resize_square(lr_img, lr_target)

            # ---- Save (동일 파일명) ----
            basename = f"{tile}-{subnum}.png"
            gt_path = gt_out_dir / basename
            lr_path = lr_out_dir / basename
            try:
                imageio.imwrite(gt_path, hr_img_resized)
                imageio.imwrite(lr_path, lr_img_resized)
            except Exception as e:
                tqdm.write(f"[SAVE-ERR] {tile}-{subnum}: {e}")
                continue

            saved_here += 1
            counts[split_dir] = counts.get(split_dir, 0) + 1

        if saved_here > 0:
            tiles_with_zero += 1

        pbar.update(1)
        pbar.set_postfix_str(
            f"tile={tile} split={split_dir} saved={saved_here} zeros={'Y' if saved_here>0 else 'N'}"
        )

    pbar.close()
    counts["tiles_with_zero"] = tiles_with_zero
    return counts

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Build (GT, LR) pairs for ALL zero-cloud revisits with tqdm & resize (HR=600, LR=150)."
    )
    ap.add_argument("--hr_base", type=str, default=str(HR_BASE_DIR), help="HR base directory")
    ap.add_argument("--lr_base", type=str, default=str(LR_BASE_DIR), help="LR base directory")
    ap.add_argument("--csv", type=str, default=str(TILE_INFO_CSV), help="Path to version2_cleaned.csv")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Output root; saves to <out>/<split>/{GT,LR}/<tile>-<sub>.png")
    ap.add_argument("--band_order", type=str, default="3,2,1",
                    help="0-indexed LR band order as 'R,G,B' (e.g., '3,2,1')")
    ap.add_argument("--hr_size", type=int, default=HR_TARGET, help="HR output size (default 600)")
    ap.add_argument("--lr_size", type=int, default=LR_TARGET, help="LR output size (default 150)")
    return ap.parse_args()

def main():
    args = parse_args()
    hr_base = Path(args.hr_base)
    lr_base = Path(args.lr_base)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    if not hr_base.exists():
        print(f"[FATAL] HR base not found: {hr_base}")
        sys.exit(1)
    if not lr_base.exists():
        print(f"[FATAL] LR base not found: {lr_base}")
        sys.exit(1)
    if not csv_path.exists():
        print(f"[FATAL] CSV not found: {csv_path}")
        sys.exit(1)

    try:
        band_order = tuple(int(x) for x in args.band_order.split(","))
        assert len(band_order) == 3
    except Exception:
        print("[FATAL] --band_order must look like '3,2,1'")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = build_pairs_zero_cloud_all_revisits(
        csv_path=csv_path,
        hr_base=hr_base,
        lr_base=lr_base,
        output_dir=output_dir,
        band_order=band_order,
        hr_target=int(args.hr_size),
        lr_target=int(args.lr_size),
    )

    print("\n=== DONE (cloud==0, all revisits) ===")
    for k in ("train", "val", "test", "other"):
        if k in stats:
            print(f"{k:>5}: {stats[k]} pairs")
    if "tiles_with_zero" in stats:
        print(f"tiles_with_zero: {stats['tiles_with_zero']} tiles had at least one zero-cloud revisit")

if __name__ == "__main__":
    main()
