# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:59:42 2026

@author: jb00202
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.signal.windows import hann, tukey
from numpy.fft import fft2, fftshift
from scipy.stats import linregress
import shapefile
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import os
from matplotlib.ticker import MultipleLocator
import pandas as pd

AXIS_FONT_SIZE = 14

plt.rcParams['font.family'] = 'Arial Black'
plt.rcParams['font.weight'] = 'black'
plt.rcParams['axes.titleweight'] = 'black'
plt.rcParams['axes.labelweight'] = 'black'

plt.rcParams['axes.titlesize'] = AXIS_FONT_SIZE
plt.rcParams['axes.labelsize'] = AXIS_FONT_SIZE
plt.rcParams['xtick.labelsize'] = AXIS_FONT_SIZE
plt.rcParams['ytick.labelsize'] = AXIS_FONT_SIZE
plt.rcParams['legend.fontsize'] = AXIS_FONT_SIZE
plt.rcParams['figure.titlesize'] = AXIS_FONT_SIZE

tif_file = "aeromag anomaly data input"

data = tiff.imread(tif_file).astype(float)
data = data.astype(np.float32, copy=False)
data[data < -1e30] = np.nan

x_scale, y_scale = 1312.6780169996309, -1312.6780169996309
x_ul = -7486754.257455864
y_ul =  11655241.972525815

E_min = 229578.62998964265 - 270000
E_max = 815507.8015687389  + 300000
N_min = 3988111.9623426683 - 80000
N_max = 4572419.9078075355 + 200000

col_min = int(np.floor((E_min - x_ul) / x_scale))
col_max = int(np.ceil ((E_max - x_ul) / x_scale))
row_min = int(np.floor((N_max - y_ul) / y_scale))
row_max = int(np.ceil ((N_min - y_ul) / y_scale))

r0, r1 = sorted([row_min, row_max])
c0, c1 = sorted([col_min, col_max])
r0 = max(r0, 0);  c0 = max(c0, 0)
r1 = min(r1, data.shape[0]);  c1 = min(c1, data.shape[1])

clip = data[r0:r1, c0:c1]
print(f"Clipped raster shape: {clip.shape}  ({clip.shape[1]} cols × {clip.shape[0]} rows)")

pixel_size_km = abs(x_scale) / 1000.0

WINDOW_KM_LIST      = [250]
MIN_PTS_FIT_ZC_LIST = [3]

UC_HEIGHT_KM = 5.0

RUN_UC_CORR_TEST = False
UC_TEST_HEIGHTS_KM = list(np.arange(0.0, 50.0, 1.0))
UC_CORR_USE_ORIGINAL_FINITE_MASK = True

overlap_frac = 0.80

SAVE_SPECTRAL_PROFILES  = False
SPECTRAL_PROFILE_DIR    = os.path.join(
    "Spectral Profile Save Output"
)
PLOT_SPECTRAL_SUMMARY   = True

PLOT_EACH_WINDOW_SPECTRUM = False
SAVE_EACH_WINDOW_SPECTRUM = False   
SPECTRUM_OUT_DIR          = SPECTRAL_PROFILE_DIR
SPECTRUM_DPI              = 150
SPECTRUM_SHOW_MAX         = None
INCLUDE_BAND_DIAGNOSTICS_IN_TITLE = True


KMIN_C = 0.05
KMAX_C = 0.14
KMIN_T, KMAX_T = 0.25, 0.65

MIN_PTS_FIT_ZT = 8

R2_MIN                 = 0.85
MIN_FINITE_WINDOW_FRAC = 0.95

ZB_MAX_KM  = 100.0
ZC_ZT_MIN  = 2.0

USE_HANN          = True
TAPER_TUKEY_ALPHA = 0.0

PAD_FACTOR = 1

MAX_MONOTONIC_DROP = 0

shp_path = "USA State Shapefile"
OUT_DIR  = "Output Directory"

APPLY_ATLANTIC_OCEAN_MASK = True
ATLANTIC_Y_BIN_KM = 10.0
ATLANTIC_BUFFER_KM = 20.0

USE_EXCEL_BAND_OVERRIDES = True
EXCEL_BANDS_PATH = "Input for User defined Bands for each spectral window"

USE_SLOPE_STABILITY = False
SLOPE_STAB_APPLY_TO = "zt"   # "zc", "zt", or "both"

ZC_SEARCH_KMIN = 0.05
ZC_SEARCH_KMAX = 0.15
ZT_SEARCH_KMIN = 0.25
ZT_SEARCH_KMAX = 0.80

ZC_BAND_NPTS_RANGE = (3, 6)
ZT_BAND_NPTS_RANGE = (8, 30)

STABILITY_SHIFT_BINS = 5


def fill_nans_mean(arr):
    arr = np.array(arr, dtype=float, copy=True)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return arr
    if not finite_mask.all():
        arr[~finite_mask] = np.nanmean(arr)
    return arr

def apply_hann_window(arr):
    wy = hann(arr.shape[0], sym=False)
    wx = hann(arr.shape[1], sym=False)
    return arr * np.outer(wy, wx)

def apply_tukey_window(arr, alpha=0.4):
    wy = tukey(arr.shape[0], alpha=alpha)
    wx = tukey(arr.shape[1], alpha=alpha)
    return arr * np.outer(wy, wx)

def zero_pad_to_power_of_two(arr, pad_factor=1):
    if pad_factor is None:
        pad_factor = 1
    nrows, ncols = arr.shape
    p2r = 2 ** int(np.ceil(np.log2(nrows)))
    p2c = 2 ** int(np.ceil(np.log2(ncols)))
    tr  = max(int(np.ceil(pad_factor * p2r)), nrows)
    tc  = max(int(np.ceil(pad_factor * p2c)), ncols)
    py, px = tr - nrows, tc - ncols
    return np.pad(arr,
                  ((py//2, py - py//2), (px//2, px - px//2)),
                  mode="constant")

def upward_continue_spatial(arr, pixel_size_km, height_km):
    if height_km <= 0:
        return arr

    ny, nx = arr.shape
    ft = np.fft.fftshift(np.fft.fft2(arr))

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_size_km)) * 2.0 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_size_km)) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    operator = np.exp(-K * height_km)
    ft_uc = ft * operator

    arr_uc = np.real(np.fft.ifft2(np.fft.ifftshift(ft_uc)))
    return arr_uc

def radial_average(power2d, pixel_size_km):
    ny, nx = power2d.shape
    N  = min(nx, ny)
    dk = 2.0 * np.pi / (N * pixel_size_km)
    y, x = np.indices(power2d.shape)
    r = np.sqrt((x - (nx-1)/2)**2 + (y - (ny-1)/2)**2).astype(int)
    tbin = np.bincount(r.ravel(), power2d.ravel())
    nr   = np.bincount(r.ravel())
    Pk   = tbin / np.maximum(nr, 1)
    k    = dk * np.arange(len(Pk), dtype=float)
    return k[1:], Pk[1:]

def drop_nonmonotonic_lowk(k, Pk, max_drop=5):
    mask = np.ones(len(k), dtype=bool)
    for _ in range(min(max_drop, len(k) - 2)):
        valid = np.where(mask)[0]
        if len(valid) < 2:
            break
        if Pk[valid[0]] < Pk[valid[1]]:
            mask[valid[0]] = False
        else:
            break
    return mask

def linear_fit_ols(k, y, kmin, kmax, min_pts=8):
    m = (k >= kmin) & (k <= kmax) & np.isfinite(y) & np.isfinite(k)
    if m.sum() < min_pts:
        return None
    slope, intercept, r, _, _ = linregress(k[m], y[m])
    return slope, intercept, r**2, int(m.sum())

def detrend_planar(arr):
    arr = np.array(arr, dtype=float, copy=True)
    nan_mask = ~np.isfinite(arr)
    finite = ~nan_mask
    if finite.mean() < 0.05:
        return arr
    rows, cols = np.indices(arr.shape)
    X = np.c_[rows[finite].ravel(), cols[finite].ravel(), np.ones(finite.sum())]
    yv = arr[finite].ravel()
    try:
        params, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        plane = params[0]*rows + params[1]*cols + params[2]
        arr = arr - plane
        arr[nan_mask] = np.nan
    except np.linalg.LinAlgError:
        pass
    return arr

# EXCEL BAND OVERRIDES
def load_excel_band_overrides(path):
    if not (path and os.path.exists(path)):
        print(f"[Excel] Band file not found: {path}")
        return {}, False

    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    needed_any = {"x0", "y0"}
    if not needed_any.issubset(set(df.columns)):
        print(f"[Excel] Missing required columns. Need at least: {needed_any}. Found: {list(df.columns)}")
        return {}, False

    has_winkm = "window_km" in df.columns
    out = {}

    def getv(row, name):
        if name not in df.columns:
            return None
        v = row[name]
        if pd.isna(v):
            return None
        return float(v)

    for _, row in df.iterrows():
        try:
            x0 = int(row["x0"])
            y0 = int(row["y0"])
        except Exception:
            continue

        if has_winkm and not pd.isna(row["window_km"]):
            key = (float(row["window_km"]), x0, y0)
        else:
            key = (x0, y0)

        out[key] = {
            "kmin_c": getv(row, "kmin_c"),
            "kmax_c": getv(row, "kmax_c"),
            "kmin_t": getv(row, "kmin_t"),
            "kmax_t": getv(row, "kmax_t"),
        }

    print(f"[Excel] Loaded {len(out)} band override rows from: {path}")
    return out, True


# SLOPE-STABILITY BAND PICKER
def _fit_slope_for_index_band(k, y, i0, i1, r2_min):
    if i1 <= i0:
        return None
    kk = k[i0:i1]
    yy = y[i0:i1]
    if kk.size < 2:
        return None
    slope, intercept, r, _, _ = linregress(kk, yy)
    r2 = r**2
    if not np.isfinite(slope) or not np.isfinite(r2):
        return None
    if r2 < r2_min:
        return None
    return slope, intercept, r2, kk.size

def pick_band_slope_stability(k, y, kmin_search, kmax_search,
                              npts_min, npts_max,
                              r2_min,
                              stability_shift_bins=2,
                              require_negative_slope=True):
    
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(k) & np.isfinite(y) & (k >= kmin_search) & (k <= kmax_search)
    idx = np.where(m)[0]
    if idx.size < max(npts_min, 2):
        return None

    best = None
    best_score = np.inf

    nmin = int(max(npts_min, 2))
    nmax = int(max(npts_max, nmin))

    for n in range(nmin, nmax + 1):
        for start_pos in range(0, idx.size - n + 1):
            band_idx = idx[start_pos:start_pos + n]
            i0 = band_idx[0]
            i1 = band_idx[-1] + 1

            fit0 = _fit_slope_for_index_band(k, y, i0, i1, r2_min)
            if fit0 is None:
                continue

            slope0, intercept0, r2_0, n0 = fit0
            if require_negative_slope and slope0 >= 0:
                continue

            slopes = []
            for ds0 in range(-stability_shift_bins, stability_shift_bins + 1):
                for ds1 in range(-stability_shift_bins, stability_shift_bins + 1):
                    j0 = i0 + ds0
                    j1 = i1 + ds1
                    if j0 < idx[0] or j1 > (idx[-1] + 1):
                        continue
                    if (j1 - j0) < nmin:
                        continue
                    fit = _fit_slope_for_index_band(k, y, j0, j1, r2_min)
                    if fit is None:
                        continue
                    s, _, _, _ = fit
                    if require_negative_slope and s >= 0:
                        continue
                    slopes.append(s)

            if len(slopes) < 3:
                continue

            slope_std = float(np.std(slopes))

            score = slope_std - 0.01 * r2_0 - 1e-5 * n0
            if score < best_score:
                best_score = score
                best = {
                    "kmin": float(k[i0]),
                    "kmax": float(k[i1 - 1]),
                    "slope": float(slope0),
                    "intercept": float(intercept0),
                    "r2": float(r2_0),
                    "n": int(n0),
                    "slope_std": float(slope_std),
                }

    return best


# TANAKA ESTIMATOR (fixed bands + optional slope-stability picks)

def estimate_curie_depth_tanaka(k, Pk,
                                kmin_c, kmax_c,
                                kmin_t=0.25, kmax_t=0.75,
                                min_pts_zc=3,
                                min_pts_zt=8,
                                r2_min=0.85,
                                max_monotonic_drop=3,
                                zb_max_km=70.0,
                                zc_zt_min=2.0,
                                use_slope_stability=False,
                                slope_stab_apply_to="both"):
    mask = (Pk > 0) & (k > 0) & np.isfinite(Pk) & np.isfinite(k)
    mono_mask      = drop_nonmonotonic_lowk(k, Pk, max_drop=max_monotonic_drop)
    mask           = mask & mono_mask
    n_dropped_mono = int((~mono_mask).sum())

    k_use = k[mask]
    P_use = Pk[mask]

    if k_use.size < (min_pts_zc + min_pts_zt):
        return None, None, None

    y_t = 0.5 * np.log(P_use)
    y_c = 0.5 * np.log(P_use) - np.log(k_use)

    kmin_c_eff, kmax_c_eff = float(kmin_c), float(kmax_c)
    kmin_t_eff, kmax_t_eff = float(kmin_t), float(kmax_t)

    picked_c = None
    picked_t = None

    if use_slope_stability:
        do_zc = (slope_stab_apply_to in ("zc", "both"))
        do_zt = (slope_stab_apply_to in ("zt", "both"))

        if do_zc:
            picked_c = pick_band_slope_stability(
                k_use, y_c,
                kmin_search=float(ZC_SEARCH_KMIN),
                kmax_search=float(ZC_SEARCH_KMAX),
                npts_min=int(ZC_BAND_NPTS_RANGE[0]),
                npts_max=int(ZC_BAND_NPTS_RANGE[1]),
                r2_min=float(r2_min),
                stability_shift_bins=int(STABILITY_SHIFT_BINS),
                require_negative_slope=True
            )
            if picked_c is not None:
                kmin_c_eff, kmax_c_eff = picked_c["kmin"], picked_c["kmax"]

        if do_zt:
            picked_t = pick_band_slope_stability(
                k_use, y_t,
                kmin_search=float(ZT_SEARCH_KMIN),
                kmax_search=float(ZT_SEARCH_KMAX),
                npts_min=int(ZT_BAND_NPTS_RANGE[0]),
                npts_max=int(ZT_BAND_NPTS_RANGE[1]),
                r2_min=float(r2_min),
                stability_shift_bins=int(STABILITY_SHIFT_BINS),
                require_negative_slope=True
            )
            if picked_t is not None:
                kmin_t_eff, kmax_t_eff = picked_t["kmin"], picked_t["kmax"]

    fit_c = linear_fit_ols(k_use, y_c, kmin_c_eff, kmax_c_eff, min_pts=min_pts_zc)
    if fit_c is None:
        return None, None, None
    slope_c, intercept_c, r2_c, n_c = fit_c
    if slope_c >= 0 or r2_c < r2_min:
        return None, None, None
    Zc = -slope_c

    fit_t = linear_fit_ols(k_use, y_t, kmin_t_eff, kmax_t_eff, min_pts=min_pts_zt)
    if fit_t is None:
        return None, None, None
    slope_t, intercept_t, r2_t, n_t = fit_t
    if slope_t >= 0 or r2_t < r2_min:
        return None, None, None
    Zt = -slope_t

    Zb = 2.0 * Zc - Zt

    if Zb > zb_max_km:
        return None, None, None
    if (Zc - Zt) < zc_zt_min:
        return None, None, None
    if Zb <= 0:
        return None, None, None

    meta = {
        "Zc_kmin":        float(kmin_c_eff),
        "Zc_kmax":        float(kmax_c_eff),
        "Zc_slope":       float(slope_c),
        "Zc_intercept":   float(intercept_c),
        "Zc_r2":          float(r2_c),
        "Zc_n":           int(n_c),
        "Zt_kmin":        float(kmin_t_eff),
        "Zt_kmax":        float(kmax_t_eff),
        "Zt_slope":       float(slope_t),
        "Zt_intercept":   float(intercept_t),
        "Zt_r2":          float(r2_t),
        "Zt_n":           int(n_t),
        "n_dropped_mono": int(n_dropped_mono),
    }

    meta["Zc_slope_std"] = float(picked_c["slope_std"]) if picked_c is not None else np.nan
    meta["Zt_slope_std"] = float(picked_t["slope_std"]) if picked_t is not None else np.nan

    return (Zt, Zc, Zb), (k_use, y_c, y_t), meta


# SPECTRAL PROFILE PLOTS

def plot_spectral_profile(k_use, y_c, y_t,
                          kmin_c, kmax_c, kmin_t, kmax_t,
                          slope_c, intercept_c,
                          slope_t, intercept_t,
                          Zt, Zc, Zb,
                          r2_c, r2_t,
                          title="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if title:
        fig.suptitle(title, fontsize=14, fontfamily='Arial Black', fontweight='black')

    ax = axes[0]
    ax.plot(k_use, y_c, 'k.', ms=3, label="ln(√P / k)")
    ax.axvspan(kmin_c, kmax_c, alpha=0.15, color='blue',
               label=f"Zc band [{kmin_c:.3f}, {kmax_c:.3f}]")
    k_fit_c = np.linspace(kmin_c, kmax_c, 100)
    ax.plot(k_fit_c, slope_c * k_fit_c + intercept_c, 'b-', lw=2,
            label=f"Zc fit: slope={slope_c:.2f}\nZc={Zc:.1f} km, R²={r2_c:.3f}")
    ax.set_xlabel("Wavenumber k (rad/km)")
    ax.set_ylabel("ln(√P / k)")
    ax.set_title("Centroid depth (Zc) spectrum")
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(10, 20)
    ax.tick_params(axis='x', labelrotation=90)

    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    ax.grid(which='major', axis='x', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k_use, y_t, 'k.', ms=3, label="ln(√P)")
    ax.axvspan(kmin_t, kmax_t, alpha=0.15, color='red',
               label=f"Zt band [{kmin_t:.3f}, {kmax_t:.3f}]")
    k_fit_t = np.linspace(kmin_t, kmax_t, 100)
    ax.plot(k_fit_t, slope_t * k_fit_t + intercept_t, 'r-', lw=2,
            label=f"Zt fit: slope={slope_t:.2f}\nZt={Zt:.1f} km, R²={r2_t:.3f}")
    ax.set_xlabel("Wavenumber k (rad/km)")
    ax.set_ylabel("ln(√P)")
    ax.set_title(f"Top depth (Zt) spectrum  |  Zb = {Zb:.1f} km")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_spectral_summary(spectral_records, window_km, min_pts_zc,
                          kmin_c, kmax_c, kmin_t, kmax_t):
    if not spectral_records:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Spectral Profile Summary  |  Window={window_km} km, min_pts_Zc={min_pts_zc}, "
        f"UC={UC_HEIGHT_KM} km  ({len(spectral_records)} windows)",
        fontsize=14, fontfamily='Arial Black', fontweight='black'
    )

    for rec in spectral_records:
        k_use = rec["k_use"]
        axes[0].plot(k_use, rec["y_c"], alpha=0.25, lw=0.6, color='steelblue')
        axes[1].plot(k_use, rec["y_t"], alpha=0.25, lw=0.6, color='tomato')

    for ax, band, color, ylabel, title_str in [
        (axes[0], (kmin_c, kmax_c), 'blue',  "ln(√P / k)", "Centroid (Zc) — all windows"),
        (axes[1], (kmin_t, kmax_t), 'red',   "ln(√P)",     "Top (Zt) — all windows"),
    ]:
        ax.axvspan(band[0], band[1], alpha=0.12, color=color,
                   label=f"fit band [{band[0]:.3f}, {band[1]:.3f}]")
        ax.set_xlabel("Wavenumber k (rad/km)")
        ax.set_ylabel(ylabel)
        ax.set_title(title_str)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def output_window_spectrum(x0, y0, window_km, label,
                           k_use, y_c, y_t,
                           meta, Zt, Zc, Zb,
                           band_src="",
                           out_dir=None,
                           do_show=True,
                           do_save=False,
                           dpi=150):

    kmin_c = float(meta["Zc_kmin"]); kmax_c = float(meta["Zc_kmax"])
    kmin_t = float(meta["Zt_kmin"]); kmax_t = float(meta["Zt_kmax"])

    title = f"win{window_km}km  x0={x0} y0={y0}  Zt={Zt:.1f}  Zc={Zc:.1f}  Zb={Zb:.1f} km"
    if INCLUDE_BAND_DIAGNOSTICS_IN_TITLE:
        zc_std = meta.get("Zc_slope_std", np.nan)
        zt_std = meta.get("Zt_slope_std", np.nan)
        title += f"\nZc[{kmin_c:.3f},{kmax_c:.3f}]  Zt[{kmin_t:.3f},{kmax_t:.3f}]  src={band_src}"
        if np.isfinite(zc_std) or np.isfinite(zt_std):
            title += f"  |  σ_slope(Zc)={zc_std:.3g}  σ_slope(Zt)={zt_std:.3g}"

    save_path = None
    if do_save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"spec_{label}_x{x0}_y{y0}.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontfamily='Arial Black', fontweight='black')

    ax = axes[0]
    ax.plot(k_use, y_c, 'k.', ms=3, label="ln(√P / k)")
    ax.axvspan(kmin_c, kmax_c, alpha=0.15, color='blue',
               label=f"Zc band [{kmin_c:.3f}, {kmax_c:.3f}]")
    k_fit_c = np.linspace(kmin_c, kmax_c, 100)
    ax.plot(k_fit_c, meta["Zc_slope"] * k_fit_c + meta["Zc_intercept"], 'b-', lw=0.5,
            label=f"Zc fit: slope={meta['Zc_slope']:.2f}\nZc={Zc:.1f} km, R²={meta['Zc_r2']:.3f}")
    ax.set_xlabel("Wavenumber k (rad/km)")
    ax.set_ylabel("ln(√P / k)")
    ax.set_title("Centroid depth (Zc) spectrum")
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(10, 20)
    ax.tick_params(axis='x', labelrotation=0)

    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    ax.grid(which='major', axis='x', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k_use, y_t, 'k.', ms=3, label="ln(√P)")
    ax.axvspan(kmin_t, kmax_t, alpha=0.15, color='red',
               label=f"Zt band [{kmin_t:.3f}, {kmax_t:.3f}]")
    k_fit_t = np.linspace(kmin_t, kmax_t, 100)
    ax.plot(k_fit_t, meta["Zt_slope"] * k_fit_t + meta["Zt_intercept"], 'r-', lw=2,
            label=f"Zt fit: slope={meta['Zt_slope']:.2f}\nZt={Zt:.1f} km, R²={meta['Zt_r2']:.3f}")
    ax.set_xlabel("Wavenumber k (rad/km)")
    ax.set_ylabel("ln(√P)")
    ax.set_title(f"Top depth (Zt) spectrum  |  Zb = {Zb:.1f} km")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=int(dpi), bbox_inches="tight")

    if do_show:
        plt.show()
    else:
        plt.close(fig)


def pearson_r_masked(a, b, mask=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    else:
        mask = mask & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan
    av = a[mask].ravel() - np.mean(a[mask])
    bv = b[mask].ravel() - np.mean(b[mask])
    denom = np.sqrt(np.sum(av**2) * np.sum(bv**2))
    if denom == 0:
        return np.nan
    return float(np.sum(av * bv) / denom)

def uc_correlation_sweep(field2d, pixel_size_km, heights_km, finite_mask=None):
    heights_km = list(heights_km)
    if len(heights_km) < 2:
        raise ValueError("heights_km must have at least 2 values.")
    field_filled = fill_nans_mean(field2d)
    prev_h  = heights_km[0]
    prev_uc = upward_continue_spatial(field_filled, pixel_size_km, prev_h)
    rows = []
    for h in heights_km[1:]:
        curr_uc = upward_continue_spatial(field_filled, pixel_size_km, h)
        r_val   = pearson_r_masked(prev_uc, curr_uc, mask=finite_mask)
        rows.append((float(prev_h), float(h),
                     float(r_val) if np.isfinite(r_val) else np.nan,
                     float(r_val**2) if np.isfinite(r_val) else np.nan))
        prev_h  = h
        prev_uc = curr_uc
    return rows


if RUN_UC_CORR_TEST:
    print("\n" + "="*65)
    print("UPWARD CONTINUATION CORRELATION SWEEP  r(h_i, h_{i+1})")
    print("="*65)
    corr_mask = np.isfinite(clip) if UC_CORR_USE_ORIGINAL_FINITE_MASK else None
    rows = uc_correlation_sweep(clip, pixel_size_km, UC_TEST_HEIGHTS_KM, corr_mask)
    print(f"{'h_i_km':>10} {'h_ip1_km':>10} {'r':>10} {'r2':>10}")
    for hi, hip1, r, r2 in rows:
        print(f"{hi:10.1f} {hip1:10.1f} {r:10.4f} {r2:10.4f}")

    try:
        plt.figure(figsize=(9, 4.5))
        hvals = [row[1] for row in rows]
        rvals = [row[2] for row in rows]
        plt.plot(hvals, rvals, "o-")
        plt.xlabel("UC height (km)")
        plt.ylabel("r(h_i, h_{i+1})")
        plt.title("Field stabilization: correlation between successive UC heights")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")

    try:
        thr = 0.98
        plateau_h = None
        for i in range(len(rows)):
            r_here = rows[i][2]
            if np.isfinite(r_here) and r_here >= thr:
                tail = [rr[2] for rr in rows[i:]]
                if np.all(np.nan_to_num(tail, nan=-1.0) >= thr):
                    plateau_h = float(rows[i][1])
                    break
        if plateau_h is not None:
            print(f"\nPlateau (r≥{thr}) begins at ~{plateau_h:.1f} km")
        else:
            print(f"\nNo clear plateau found for r≥{thr}.")
    except Exception:
        pass

def make_patches_km(sf):
    plist = []
    for sr in sf.shapeRecords():
        pts   = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            ring_km = [(x/1000.0, y/1000.0) for x, y in pts[parts[i]:parts[i+1]]]
            plist.append(Polygon(ring_km, closed=True))
    return plist

def build_coastline_proxy_from_states(sf, y_bin_km=10.0):
    xs_km = []
    ys_km = []
    for sr in sf.shapeRecords():
        pts = sr.shape.points
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)
        xs_km.append(arr[:, 0] / 1000.0)
        ys_km.append(arr[:, 1] / 1000.0)

    if not xs_km:
        raise ValueError("Shapefile has no points; cannot build coastline proxy.")

    xs_km = np.concatenate(xs_km)
    ys_km = np.concatenate(ys_km)

    y_min = float(np.nanmin(ys_km))
    y_max = float(np.nanmax(ys_km))

    y_edges = np.arange(y_min, y_max + y_bin_km, y_bin_km)
    if y_edges.size < 2:
        raise ValueError("Not enough y-range to bin coastline proxy.")

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_max = np.full_like(y_centers, np.nan, dtype=float)

    bin_idx = np.digitize(ys_km, y_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(y_centers))
    bin_idx = bin_idx[valid]
    xs_v = xs_km[valid]

    for i in range(len(y_centers)):
        m = (bin_idx == i)
        if np.any(m):
            x_max[i] = np.nanmax(xs_v[m])

    good = np.isfinite(x_max)
    if good.sum() < 2:
        raise ValueError("Could not build a stable coastline proxy (too many empty bins).")

    x_max_filled = np.interp(y_centers, y_centers[good], x_max[good])

    def coast_x_km(N_km):
        return np.interp(
            np.asarray(N_km, dtype=float),
            y_centers, x_max_filled,
            left=x_max_filled[0], right=x_max_filled[-1]
        )

    return coast_x_km

def window_hits_atlantic(x0, y0, window_px,
                         c0, r0,
                         x_ul, y_ul, x_scale, y_scale,
                         coast_x_km,
                         buffer_km=20.0,
                         n_samples=5):
    col_left  = c0 + x0
    col_right = c0 + x0 + window_px
    row_top   = r0 + y0
    row_bot   = r0 + y0 + window_px

    E_left_m  = x_ul + (col_left)  * x_scale
    E_right_m = x_ul + (col_right) * x_scale
    E_max_km  = max(E_left_m, E_right_m) / 1000.0

    N_top_m = y_ul + (row_top) * y_scale
    N_bot_m = y_ul + (row_bot) * y_scale
    N_min_km = min(N_top_m, N_bot_m) / 1000.0
    N_max_km = max(N_top_m, N_bot_m) / 1000.0

    Ns_km = np.linspace(N_min_km, N_max_km, max(int(n_samples), 2))
    coast_km = coast_x_km(Ns_km) + float(buffer_km)

    return bool(np.any(E_max_km > coast_km))

def triangulated_grid(center_coords_km, values, E_km_grid, N_km_grid):
    x = center_coords_km[:, 0]
    y = center_coords_km[:, 1]
    tri    = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(tri, values)
    Zg     = np.array(interp(E_km_grid, N_km_grid), dtype=float)
    tri_id = tri.get_trifinder()(E_km_grid, N_km_grid)
    Zg[tri_id == -1] = np.nan
    return Zg


def build_atlantic_ocean_mask(E_km, N_km, sf, y_bin_km=10.0, buffer_km=20.0):
    xs = []
    ys = []
    for sr in sf.shapeRecords():
        pts = sr.shape.points
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)
        xs.append(arr[:, 0] / 1000.0)
        ys.append(arr[:, 1] / 1000.0)

    if not xs:
        return np.zeros(E_km.shape, dtype=bool)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    y_edges = np.arange(y_min, y_max + y_bin_km, y_bin_km)
    if y_edges.size < 2:
        return np.zeros(E_km.shape, dtype=bool)

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_max = np.full(y_centers.shape, np.nan, dtype=float)

    bin_idx = np.digitize(ys, y_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(y_centers))
    bin_idx = bin_idx[valid]
    xs_v = xs[valid]

    for i in range(len(y_centers)):
        m = (bin_idx == i)
        if np.any(m):
            x_max[i] = np.nanmax(xs_v[m])

    good = np.isfinite(x_max)
    if good.sum() < 2:
        return np.zeros(E_km.shape, dtype=bool)

    x_max_filled = np.interp(y_centers, y_centers[good], x_max[good])

    N_rows = N_km[:, 0]
    coast_x_by_row = np.interp(
        N_rows, y_centers, x_max_filled,
        left=x_max_filled[0], right=x_max_filled[-1]
    )
    coast_x_by_row = coast_x_by_row + float(buffer_km)

    return (E_km > coast_x_by_row[:, None])

def plot_depth_maps(curie_depths, center_coords_m, sf,
                    window_km, min_pts_zc, label):
    if not center_coords_m:
        print(f"  [{label}] No valid data — skipping plots.")
        return

    center_coords_arr_m = np.array(center_coords_m, dtype=float)
    center_coords_km    = center_coords_arr_m / 1000.0
    cx_km = center_coords_km[:, 0]
    cy_km = center_coords_km[:, 1]

    Zb_values = np.array([d["Zb_km"] for d in curie_depths], dtype=float)
    Zt_values = np.array([d["Zt_km"] for d in curie_depths], dtype=float)
    Zc_values = np.array([d["Zc_km"] for d in curie_depths], dtype=float)

    pad_m   = 20_000
    xlim_km = ((E_min - pad_m)/1000, (E_max + pad_m)/1000)
    ylim_km = ((N_min - pad_m)/1000, (N_max + pad_m)/1000)

    E_grid_km = np.linspace(E_min/1000, E_max/1000, 200)
    N_grid_km = np.linspace(N_min/1000, N_max/1000, 200)
    E_km, N_km = np.meshgrid(E_grid_km, N_grid_km)

    x0_arr = np.array([d["x0"] for d in curie_depths], dtype=int)
    y0_arr = np.array([d["y0"] for d in curie_depths], dtype=int)

    def nearest_idx(px_km, py_km):
        d2 = (cx_km - px_km)**2 + (cy_km - py_km)**2
        return int(np.argmin(d2))

    corners = {
        "ul": (xlim_km[0], ylim_km[1]),
        "ur": (xlim_km[1], ylim_km[1]),
        "ll": (xlim_km[0], ylim_km[0]),
        "lr": (xlim_km[1], ylim_km[0]),
    }
    corner_labels = {}
    for key, (px, py) in corners.items():
        ii = nearest_idx(px, py)
        corner_labels[key] = f"x0={x0_arr[ii]}  y0={y0_arr[ii]}"

    def draw_corner_labels(ax):
        ax.text(0.01, 0.99, corner_labels["ul"], transform=ax.transAxes,
                ha="left", va="top", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
        ax.text(0.99, 0.99, corner_labels["ur"], transform=ax.transAxes,
                ha="right", va="top", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
        ax.text(0.01, 0.01, corner_labels["ll"], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
        ax.text(0.99, 0.01, corner_labels["lr"], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    def estimate_step_km_from_centers(cx, cy):
        ux = np.unique(np.round(cx, 6))
        if ux.size >= 2:
            dx = np.diff(np.sort(ux))
            dx = dx[dx > 0]
            if dx.size:
                return float(np.median(dx))
        uy = np.unique(np.round(cy, 6))
        if uy.size >= 2:
            dy = np.diff(np.sort(uy))
            dy = dy[dy > 0]
            if dy.size:
                return float(np.median(dy))
        return None

    step_km_est = estimate_step_km_from_centers(cx_km, cy_km)
    if step_km_est is None or not np.isfinite(step_km_est) or step_km_est <= 0:
        step_km_est = float(window_km * (1.0 - overlap_frac))
    half_box = 0.5 * step_km_est

    def add_step_boxes(ax, cx, cy, half, edgecolor="k", lw=0.3, alpha=0.35, zorder=2):
        for xci, yci in zip(cx, cy):
            ax.add_patch(Rectangle((xci - half, yci - half),
                                   2*half, 2*half,
                                   fill=False, edgecolor=edgecolor,
                                   linewidth=lw, alpha=alpha, zorder=zorder))

    def box_union_mask(Egrid, Ngrid, cx, cy, half):
        mask = np.zeros(Egrid.shape, dtype=bool)
        for xci, yci in zip(cx, cy):
            mask |= (np.abs(Egrid - xci) <= half) & (np.abs(Ngrid - yci) <= half)
        return mask

    boxes_mask = box_union_mask(E_km, N_km, cx_km, cy_km, half_box)

    atlantic_mask = None
    if APPLY_ATLANTIC_OCEAN_MASK:
        atlantic_mask = build_atlantic_ocean_mask(
            E_km, N_km, sf,
            y_bin_km=ATLANTIC_Y_BIN_KM,
            buffer_km=ATLANTIC_BUFFER_KM
        )

    depth_vars = [
        (Zt_values, "turbo_r", "Zt (km)", "Top Depth Zt"),
        (Zc_values, "turbo_r", "Zc (km)", "Centroid Depth Zc"),
        (Zb_values, "turbo_r", "Zb (km)", "Bottom Depth Zb"),
    ]

    for values, cmap, cbar_label, depth_title in depth_vars:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f"{depth_title}  |  Window={window_km} km, min_pts_Zc={min_pts_zc}, "
            f"UC={UC_HEIGHT_KM} km\n({len(curie_depths)} accepted windows)",
            fontsize=14, fontweight='bold'
        )

        ax = axes[0]
        sc = ax.scatter(cx_km, cy_km, c=values, cmap=cmap, s=60, marker="s", zorder=3)
        fig.colorbar(sc, ax=ax, label=cbar_label)

        add_step_boxes(ax, cx_km, cy_km, half_box)

        ax.add_collection(PatchCollection(
            make_patches_km(sf), facecolor="none", edgecolor="black", lw=1.5, zorder=4
        ))
        ax.set_xlim(*xlim_km);  ax.set_ylim(*ylim_km)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (km)");  ax.set_ylabel("Northing (km)")
        ax.set_title("Scatter Map (centers) + step boxes")
        draw_corner_labels(ax)

        ax = axes[1]
        Z_masked = triangulated_grid(center_coords_km, values, E_km, N_km)

        Z_masked[~boxes_mask] = np.nan
        if atlantic_mask is not None:
            Z_masked[atlantic_mask] = np.nan

        cf = ax.contourf(E_km, N_km, Z_masked, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=cbar_label)

        add_step_boxes(ax, cx_km, cy_km, half_box, lw=0.25, alpha=0.25, zorder=9)

        ax.add_collection(PatchCollection(
            make_patches_km(sf), facecolor="none", edgecolor="black", lw=1.5, zorder=10
        ))
        ax.set_xlim(*xlim_km);  ax.set_ylim(*ylim_km)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (km)");  ax.set_ylabel("Northing (km)")
        ax.set_title("Interpolated Contour Map\n(triangulation + box-clipped + Atlantic-masked)")
        draw_corner_labels(ax)

        plt.tight_layout()
        plt.show()

sf = shapefile.Reader(shp_path)
coast_x_km = build_coastline_proxy_from_states(sf, y_bin_km=10.0)

ATLANTIC_BUFFER_KM = 50.0
ATLANTIC_NSAMPLES  = 10

if SAVE_SPECTRAL_PROFILES:
    os.makedirs(SPECTRAL_PROFILE_DIR, exist_ok=True)

if SAVE_EACH_WINDOW_SPECTRUM:
    os.makedirs(SPECTRUM_OUT_DIR, exist_ok=True)

excel_overrides = {}
excel_ok = False
if USE_EXCEL_BAND_OVERRIDES:
    excel_overrides, excel_ok = load_excel_band_overrides(EXCEL_BANDS_PATH)

def get_bands_for_window(window_km, x0, y0):
    if USE_EXCEL_BAND_OVERRIDES and excel_ok:
        key3 = (float(window_km), int(x0), int(y0))
        key2 = (int(x0), int(y0))
        if key3 in excel_overrides:
            b = excel_overrides[key3]
            return (
                b["kmin_c"] if b["kmin_c"] is not None else KMIN_C,
                b["kmax_c"] if b["kmax_c"] is not None else KMAX_C,
                b["kmin_t"] if b["kmin_t"] is not None else KMIN_T,
                b["kmax_t"] if b["kmax_t"] is not None else KMAX_T,
                "excel(window_km,x0,y0)"
            )
        if key2 in excel_overrides:
            b = excel_overrides[key2]
            return (
                b["kmin_c"] if b["kmin_c"] is not None else KMIN_C,
                b["kmax_c"] if b["kmax_c"] is not None else KMAX_C,
                b["kmin_t"] if b["kmin_t"] is not None else KMIN_T,
                b["kmax_t"] if b["kmax_t"] is not None else KMAX_T,
                "excel(x0,y0)"
            )

    return (KMIN_C, KMAX_C, KMIN_T, KMAX_T, "fixed_or_slope_stability")

for window_km in WINDOW_KM_LIST:
    for min_pts_zc in MIN_PTS_FIT_ZC_LIST:

        label = f"win{window_km}km_minpts{min_pts_zc}_UC{UC_HEIGHT_KM}km"
        print(f"\n{'='*65}")
        print(f"ITERATION: window_km={window_km}  |  min_pts_fit_zc={min_pts_zc}  |  UC={UC_HEIGHT_KM} km")
        print(f"  Fixed fallback bands: Zc [{KMIN_C}, {KMAX_C}]  |  Zt [{KMIN_T}, {KMAX_T}] rad/km")
        print(f"  Excel overrides: {USE_EXCEL_BAND_OVERRIDES}  (loaded={excel_ok})")
        print(f"  Slope-stability: {USE_SLOPE_STABILITY}  (apply_to={SLOPE_STAB_APPLY_TO})")
        print(f"  Per-window spectrum: show={PLOT_EACH_WINDOW_SPECTRUM} save={SAVE_EACH_WINDOW_SPECTRUM}")
        print(f"{'='*65}")

        window_px = int(round(window_km / pixel_size_km))
        step_px   = max(int(round(window_px * (1.0 - overlap_frac))), 1)

        clip_h, clip_w = clip.shape
        x_starts = np.arange(0, clip_w - window_px + 1, step_px)
        y_starts = np.arange(0, clip_h - window_px + 1, step_px)

        print(f"  Window: {window_px} px  |  Step: {step_px} px  |  "
              f"Grid: {len(x_starts)} × {len(y_starts)} = {len(x_starts)*len(y_starts)} windows")

        curie_depths     = []
        spectral_records = []
        total    = 0
        rejected = {"all_nan": 0, "finite_frac": 0, "result_none": 0, "accepted": 0, "atlantic": 0}

        shown_count = 0

        for y0 in y_starts:
            for x0 in x_starts:

                if window_hits_atlantic(
                        x0, y0, window_px,
                        c0, r0,
                        x_ul, y_ul, x_scale, y_scale,
                        coast_x_km,
                        buffer_km=ATLANTIC_BUFFER_KM,
                        n_samples=ATLANTIC_NSAMPLES
                ):
                    rejected["atlantic"] += 1
                    continue

                window = clip[y0:y0+window_px, x0:x0+window_px]

                if np.isnan(window).all():
                    rejected["all_nan"] += 1
                    continue
                if np.isfinite(window).mean() < MIN_FINITE_WINDOW_FRAC:
                    rejected["finite_frac"] += 1
                    continue

                w = detrend_planar(window)                                     # 1) detrend
                w = fill_nans_mean(w)                                          # 2) fill NaNs
                w = upward_continue_spatial(w, pixel_size_km, UC_HEIGHT_KM)    # 3) UC
                w = apply_hann_window(w) if USE_HANN else \
                    apply_tukey_window(w, alpha=TAPER_TUKEY_ALPHA)             # 4) taper
                w = zero_pad_to_power_of_two(w, pad_factor=PAD_FACTOR)         # 5) pad
                ft = fftshift(fft2(w))                                         # 6) FFT
                k, Pk = radial_average(np.abs(ft)**2, pixel_size_km)           # 7) radial avg

                kmin_c_eff, kmax_c_eff, kmin_t_eff, kmax_t_eff, band_src = get_bands_for_window(window_km, x0, y0)

                allow_stab = bool(USE_SLOPE_STABILITY and band_src.startswith("fixed"))
                result, spectra, meta = estimate_curie_depth_tanaka(
                    k, Pk,
                    kmin_c             = kmin_c_eff,
                    kmax_c             = kmax_c_eff,
                    kmin_t             = kmin_t_eff,
                    kmax_t             = kmax_t_eff,
                    min_pts_zc         = min_pts_zc,
                    min_pts_zt         = MIN_PTS_FIT_ZT,
                    r2_min             = R2_MIN,
                    max_monotonic_drop = MAX_MONOTONIC_DROP,
                    zb_max_km          = ZB_MAX_KM,
                    zc_zt_min          = ZC_ZT_MIN,
                    use_slope_stability= allow_stab,
                    slope_stab_apply_to= SLOPE_STAB_APPLY_TO
                )

                if result is not None:
                    Zt, Zc, Zb = result
                    k_use, y_c, y_t = spectra
                    rejected["accepted"] += 1
                    total += 1

                    spectral_records.append({
                        "k_use": k_use,
                        "y_c":   y_c,
                        "y_t":   y_t,
                        "Zt":    Zt,
                        "Zc":    Zc,
                        "Zb":    Zb,
                    })

                    if PLOT_EACH_WINDOW_SPECTRUM or SAVE_EACH_WINDOW_SPECTRUM:
                        do_show = bool(PLOT_EACH_WINDOW_SPECTRUM)
                        if SPECTRUM_SHOW_MAX is not None:
                            do_show = do_show and (shown_count < int(SPECTRUM_SHOW_MAX))

                        output_window_spectrum(
                            x0=x0, y0=y0, window_km=window_km, label=label,
                            k_use=k_use, y_c=y_c, y_t=y_t,
                            meta=meta, Zt=Zt, Zc=Zc, Zb=Zb,
                            band_src=band_src,
                            out_dir=SPECTRUM_OUT_DIR,
                            do_show=do_show,
                            do_save=bool(SAVE_EACH_WINDOW_SPECTRUM),
                            dpi=int(SPECTRUM_DPI),
                        )
                        if do_show:
                            shown_count += 1

                    if SAVE_SPECTRAL_PROFILES:
                        sp_title = (f"win{window_km}km  x0={x0} y0={y0}  "
                                    f"Zt={Zt:.1f} Zc={Zc:.1f} Zb={Zb:.1f} km\n"
                                    f"bands: Zc[{meta['Zc_kmin']:.3f},{meta['Zc_kmax']:.3f}] "
                                    f"Zt[{meta['Zt_kmin']:.3f},{meta['Zt_kmax']:.3f}]  src={band_src}")
                        sp_path  = os.path.join(
                            SPECTRAL_PROFILE_DIR,
                            f"spec_{label}_x{x0}_y{y0}.png"
                        )
                        plot_spectral_profile(
                            k_use, y_c, y_t,
                            meta["Zc_kmin"], meta["Zc_kmax"], meta["Zt_kmin"], meta["Zt_kmax"],
                            meta["Zc_slope"], meta["Zc_intercept"],
                            meta["Zt_slope"], meta["Zt_intercept"],
                            Zt, Zc, Zb,
                            meta["Zc_r2"], meta["Zt_r2"],
                            title=sp_title, save_path=sp_path
                        )

                    curie_depths.append({
                        "x0":             int(x0),
                        "y0":             int(y0),
                        "window_km":      float(window_km),
                        "min_pts_zc":     int(min_pts_zc),
                        "uc_height_km":   float(UC_HEIGHT_KM),
                        "Zt_km":          float(Zt),
                        "Zc_km":          float(Zc),
                        "Zb_km":          float(Zb),
                        "Zc_kmin":        float(meta["Zc_kmin"]),
                        "Zc_kmax":        float(meta["Zc_kmax"]),
                        "Zt_kmin":        float(meta["Zt_kmin"]),
                        "Zt_kmax":        float(meta["Zt_kmax"]),
                        "Zc_slope":       float(meta["Zc_slope"]),
                        "Zt_slope":       float(meta["Zt_slope"]),
                        "Zc_r2":          float(meta["Zc_r2"]),
                        "Zt_r2":          float(meta["Zt_r2"]),
                        "Zc_n":           int(meta["Zc_n"]),
                        "Zt_n":           int(meta["Zt_n"]),
                        "Zc_slope_std":   float(meta.get("Zc_slope_std", np.nan)),
                        "Zt_slope_std":   float(meta.get("Zt_slope_std", np.nan)),
                        "n_dropped_mono": int(meta["n_dropped_mono"]),
                        "band_source":    str(band_src),
                    })
                else:
                    rejected["result_none"] += 1

        print(f"  Done: {total} accepted windows")
        print(f"  Rejection breakdown: {rejected}")

        if PLOT_SPECTRAL_SUMMARY and spectral_records:
            plot_spectral_summary(
                spectral_records, window_km, min_pts_zc,
                KMIN_C, KMAX_C, KMIN_T, KMAX_T
            )

        center_coords_m = []
        for item in curie_depths:
            center_row = item["y0"] + window_px // 2 + r0
            center_col = item["x0"] + window_px // 2 + c0
            E = x_ul + (center_col + 0.5) * x_scale
            N = y_ul + (center_row + 0.5) * y_scale
            center_coords_m.append((E, N))

        plot_depth_maps(curie_depths, center_coords_m, sf, window_km, min_pts_zc, label)
        
        if curie_depths and center_coords_m:
            rows = []
            for item, (E, N) in zip(curie_depths, center_coords_m):
                row = dict(item)
                row["E_m"] = float(E)
                row["N_m"] = float(N)
                rows.append(row)
            df_cpd = pd.DataFrame(rows)
            csv_path = os.path.join(OUT_DIR, f"cpd_results_{label}.csv")
            df_cpd.to_csv(csv_path, index=False)
            print(f"  Saved CPD results CSV: {csv_path}")
            

                  # Zc FIT QUALITY DIAGNOSTICS

        if curie_depths:
            df_diag = pd.DataFrame(curie_depths)

            zc_r2 = df_diag["Zc_r2"].values
            zc_n  = df_diag["Zc_n"].values
            zc_std = df_diag["Zc_slope_std"].values

            print("\n" + "="*60)
            print("Zc FIT QUALITY DIAGNOSTICS")
            print("="*60)

            print(f"Total accepted windows        : {len(zc_r2)}")

            print("\n--- R² Statistics ---")
            print(f"Mean Zc R²                   : {np.nanmean(zc_r2):.4f}")
            print(f"Median Zc R²                 : {np.nanmedian(zc_r2):.4f}")
            print(f"Minimum Zc R²                : {np.nanmin(zc_r2):.4f}")
            print(f"Maximum Zc R²                : {np.nanmax(zc_r2):.4f}")
            print(f"Std Dev Zc R²                : {np.nanstd(zc_r2):.4f}")

            r2_thresh = R2_MIN
            frac_near = np.sum((zc_r2 >= r2_thresh) & (zc_r2 < r2_thresh + 0.05)) / len(zc_r2)
            print(f"Fraction near threshold (R²<{r2_thresh+0.05:.2f}): {frac_near:.3f} ({frac_near*100:.1f}%)")

            print("\n--- Fit Robustness ---")
            print(f"Mean # points in Zc fit      : {np.nanmean(zc_n):.2f}")
            print(f"Min # points in Zc fit       : {np.nanmin(zc_n)}")
            print(f"Max # points in Zc fit       : {np.nanmax(zc_n)}")

            print("\n--- Slope Stability ---")
            if np.all(np.isnan(zc_std)):
                print("Slope stability not used (all NaN)")
            else:
                print(f"Mean slope std (Zc)          : {np.nanmean(zc_std):.4f}")
                print(f"Median slope std (Zc)        : {np.nanmedian(zc_std):.4f}")
                print(f"Max slope std (Zc)           : {np.nanmax(zc_std):.4f}")

            try:
                plt.figure()
                plt.hist(zc_r2, bins=30)
                plt.xlabel("Zc R²")
                plt.ylabel("Frequency")
                plt.title("Distribution of Zc R² Values")
                plt.grid(True)
                plt.show()
            except Exception:
                pass

            print("="*60 + "\n")
            
            if curie_depths:
                df_diag = pd.DataFrame(curie_depths).copy()

                if "E_m" not in df_diag.columns or "N_m" not in df_diag.columns:
                    center_row = df_diag["y0"].to_numpy(dtype=float) + window_px // 2 + r0
                    center_col = df_diag["x0"].to_numpy(dtype=float) + window_px // 2 + c0
                    df_diag["E_m"] = x_ul + (center_col + 0.5) * x_scale
                    df_diag["N_m"] = y_ul + (center_row + 0.5) * y_scale

                df_worst = df_diag.sort_values(by="Zc_r2", ascending=True).head(5)

                print("\n" + "="*60)
                print("5 LOWEST Zc R² FIT LOCATIONS")
                print("="*60)

                for rank, (_, row) in enumerate(df_worst.iterrows(), start=1):
                    print(
            f"\nRank       : {rank}\n"
            f"Zc R²      : {row['Zc_r2']:.4f}\n"
            f"Zc (km)    : {row['Zc_km']:.2f}\n"
            f"Zb (km)    : {row['Zb_km']:.2f}\n"
            f"Zt (km)    : {row['Zt_km']:.2f}\n"
            f"# points   : {int(row['Zc_n'])}\n"
            f"x0, y0     : ({int(row['x0'])}, {int(row['y0'])})\n"
            f"E, N (km)  : ({row['E_m']/1000:.2f}, {row['N_m']/1000:.2f})\n"
            f"Band       : [{row['Zc_kmin']:.3f}, {row['Zc_kmax']:.3f}]\n"
            f"Source     : {row['band_source']}"
        )

                print("\n" + "="*60 + "\n")
print("\nAll iterations complete.")

