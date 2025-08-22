#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute mean, std, min, max for 18 channels across all NPZ samples.

ERA5 (13):
    z500, z850, t500, t850, u500, u850, v500, v850,
    t2m, u10, v10, mslp, tcwv

WRF  (5):
    my_output level 0–4   (rename in wrf_names[] if需要)

Each NPZ must contain:
    era   → (13, Ny, Nx)
    hirez → (5,  NyH, NxH)

NaN 值被排除在统计之外，但会计数。
"""

import os
import glob
import numpy as np

# ───────── 用户可调 ───────── #
NPZ_DIR = "/public/home/niuzeyi/down2/traindata"
# ─────────────────────────── #

# 仅用于打印
era_names = [
    "ERA_z500","ERA_z850","ERA_t500","ERA_t850","ERA_u500","ERA_u850","ERA_v500","ERA_v850",
    "ERA_t2m","ERA_u10","ERA_v10","ERA_mslp","ERA_tcwv"
]
wrf_names = [f"WRF_{i+1}" for i in range(5)]   # 如果有真实名称可替换
all_names = era_names + wrf_names              # 18

def stats_18var(npz_dir: str):
    n_ch = 18
    s1   = np.zeros(n_ch, dtype=np.float64)   # Σx
    s2   = np.zeros(n_ch, dtype=np.float64)   # Σx²
    cnt  = np.zeros(n_ch, dtype=np.int64)     # ΣN
    vmin = np.full(n_ch,  np.inf,  dtype=np.float64)
    vmax = np.full(n_ch, -np.inf,  dtype=np.float64)
    nan_cnt = np.zeros(n_ch, dtype=np.int64)

    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    print(f"Found {len(files)} npz files in {npz_dir}")
    if not files:
        return None

    expected_shape = {}

    for i, f in enumerate(files, 1):
        try:
            dat = np.load(f)

            # -- 键检查 --
            if not {"era", "hirez"}.issubset(dat.files):
                print(f"Error in {f}: missing 'era' or 'hirez'")
                continue

            era   = dat["era"]     # (13, Ny, Nx)
            hirez = dat["hirez"]   # (5,  NyH, NxH)

            # -- 形状检查 --
            if i == 1:
                expected_shape["era"]   = era.shape
                expected_shape["hirez"] = hirez.shape
                if era.shape[0] != 13 or hirez.shape[0] != 5:
                    print(f"Error in {f}: expected 13+5 channels, got era={era.shape[0]}, hirez={hirez.shape[0]}")
                    continue
            else:
                if era.shape != expected_shape["era"] or hirez.shape != expected_shape["hirez"]:
                    print(f"Error in {f}: shape mismatch. "
                          f"Expected era={expected_shape['era']}, hirez={expected_shape['hirez']}, "
                          f"got era={era.shape}, hirez={hirez.shape}")
                    continue

            # -------- ERA5 (0–12) --------
            for k in range(13):
                idx   = k
                vec   = era[k].ravel()
                mask  = np.isnan(vec)
                nan_cnt[idx] += mask.sum()

                valid = vec[~mask]
                if valid.size:
                    s1[idx]  += valid.sum()
                    s2[idx]  += np.square(valid).sum()
                    cnt[idx] += valid.size
                    vmin[idx] = min(vmin[idx], valid.min())
                    vmax[idx] = max(vmax[idx], valid.max())

            # -------- WRF (13–17) --------
            for k in range(5):
                idx   = 13 + k
                vec   = hirez[k].ravel()
                mask  = np.isnan(vec)
                nan_cnt[idx] += mask.sum()

                valid = vec[~mask]
                if valid.size:
                    s1[idx]  += valid.sum()
                    s2[idx]  += np.square(valid).sum()
                    cnt[idx] += valid.size
                    vmin[idx] = min(vmin[idx], valid.min())
                    vmax[idx] = max(vmax[idx], valid.max())

            if i % 10 == 0 or i == len(files):
                print(f"  processed {i}/{len(files)}")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    if np.any(cnt == 0):
        print("Error: one or more channels have no valid data.")
        return None

    mean = s1 / cnt
    std  = np.sqrt(s2 / cnt - mean**2)
    return mean, std, vmin, vmax, nan_cnt

# ────────── 主程序 ──────────
if __name__ == "__main__":
    res = stats_18var(NPZ_DIR)
    if res is None:
        print("No valid statistics computed.")
        exit(1)

    mean, std, vmin, vmax, nan_cnt = res
    print("\n=== Statistics for 18 variables ===")
    for i, n in enumerate(all_names):
        print(f"{n:12s}: "
              f"mean = {mean[i]:12.4f}  "
              f"std = {std[i]:10.4f}  "
              f"min = {vmin[i]:10.4f}  "
              f"max = {vmax[i]:10.4f}  "
              f"NaN = {nan_cnt[i]}")
