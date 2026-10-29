"""Standalone Computer-Vision Module for Progressive Circle, Object-Outline & Indicator Line Drawing Detection.

Detects animations that progressively draw:
1. `circle_draw` — Circles, ovals, reticles, or rings (>= 225 deg) drawn around an object to highlight/indicate it.
2. `contour_outline_draw` — Bounding boxes, polygons, or irregular contour lines drawn around an object/region.
3. `callout_line_draw` — Thin leader, pointer, or L-shaped callout lines extending to point at or label an object.

Pinpoints both the exact drawing `start_s` and exact `finish_s` (`end_s`) plus the frame-by-frame
`draw_curve` in [0.0, 1.0], designed as an independent, opt-in module (`python -m pyxelate.stroke_draws`
or MCP tool `video_detect_stroke_draws`) so scripts can inspect or selectively map indicator drawings
to musical song parts without forcing every visual trigger onto a video.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    label,
    uniform_filter1d,
)
import imageio.v3 as iio

from .fast_ops import resize_float, rgb2hsv_fast


def _spearman_corr(arr: np.ndarray) -> float:
    """Fast Spearman rank correlation of `arr` against `0..N-1` in [-1, 1]."""
    n = len(arr)
    if n < 5 or float(np.std(arr)) < 1e-6:
        return 0.0
    ranks = np.argsort(np.argsort(arr)).astype(np.float64)
    t = np.arange(n, dtype=np.float64)
    rc = ranks - np.mean(ranks)
    tc = t - np.mean(t)
    denom = float(np.sqrt(np.sum(rc * rc) * np.sum(tc * tc)))
    return float(np.sum(rc * tc) / denom) if denom > 1e-9 else 0.0


def _classify_stroke_geometry(
    ys: np.ndarray,
    xs: np.ndarray,
    birth_frames: np.ndarray,
    end_rgb_frame: np.ndarray,
    work_h: int,
    work_w: int,
) -> Optional[Dict[str, Any]]:
    """Classifies a verified persistent progressive thin stroke into circle, contour outline, or callout line."""
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    area = len(ys)
    diag = float(np.hypot(bw, bh))
    min_dim = float(min(work_h, work_w))

    pad = 5
    py0, py1 = max(0, y0 - pad), min(work_h, y1 + pad)
    px0, px1 = max(0, x0 - pad), min(work_w, x1 + pad)
    local_mask = np.zeros((py1 - py0, px1 - px0), dtype=bool)
    local_mask[ys - py0, xs - px0] = True

    # Check single-stroke connectivity (1px dilation bridges anti-aliased thin diagonals)
    dil = binary_dilation(local_mask, iterations=1)
    lbl_d, n_d = label(dil)
    if n_d > 1:
        counts = [int(np.sum(lbl_d[local_mask] == k)) for k in range(1, n_d + 1)]
        if max(counts) < 0.78 * area:
            return None

    # Measure stroke thickness via Euclidean distance transform
    dt = distance_transform_edt(dil) - 0.8
    mean_radius = float(np.mean(np.maximum(0.5, dt[local_mask])))
    max_radius = float(np.max(dt[local_mask]))
    if mean_radius > 2.45 or max_radius > 4.8:
        return None

    # Reject pure horizontal text-caption lines in the top HUD header band
    if y1 <= int(work_h * 0.16) and bw > bh * 3.0:
        return None

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    dy = ys.astype(np.float64) - cy
    dx = xs.astype(np.float64) - cx
    angles = np.arctan2(dy, dx)  # [-pi, pi]
    sector_bins = np.floor((angles + np.pi) / (2.0 * np.pi) * 24.0).astype(int) % 24
    angular_coverage_deg = int(len(np.unique(sector_bins))) * 15

    # Two-Sided Ridge Prominence Check:
    # A genuine drawn line/circle/outline stands out in RGB from its immediate neighborhood
    # (allowing slightly lower inner contrast >= 0.52 when a closed/arc circle hugs a bright object,
    # while requiring >= 0.72 for straight lines so sliding metal collimator blades are rejected!).
    outer_ring = binary_dilation(dil, iterations=3) & (~dil)
    if int(np.sum(outer_ring)) >= 8:
        stroke_rgb = np.median(end_rgb_frame[ys, xs, :], axis=0)
        patch_rgb = end_rgb_frame[py0:py1, px0:px1, :]
        outer_diffs = np.max(np.abs(patch_rgb[outer_ring] - stroke_rgb[None, :]), axis=-1)
        ridge_distinct_frac = float(np.mean(outer_diffs > 0.12))
        min_ridge = 0.52 if angular_coverage_deg >= 195 else 0.72
        if ridge_distinct_frac < min_ridge:
            return None

    # Hollow interior / enclosed object check
    filled = binary_fill_holes(dil)
    hole_mask = filled & (~dil)
    hole_area = int(np.sum(hole_mask))
    bbox_area = float(max(1, bh * bw))
    fill_density = float(area) / bbox_area

    # Ellipse-normalized radial coefficient of variation (distinguishes circles/ovals from polygons/contours)
    rx = max(2.5, float(np.std(dx)) * np.sqrt(2.0))
    ry = max(2.5, float(np.std(dy)) * np.sqrt(2.0))
    r_ell = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    ellipse_cv = float(np.std(r_ell) / max(1e-4, np.mean(r_ell)))

    # Order pixels by birth frame to inspect moving pen-tip trajectory & angular sweep direction!
    order = np.argsort(birth_frames)
    n_q = max(2, min(8, area // 22))
    start_xy = (float(np.mean(xs[order[:n_q]])), float(np.mean(ys[order[:n_q]])))
    end_xy = (float(np.mean(xs[order[-n_q:]])), float(np.mean(ys[order[-n_q:]])))

    aspect = float(bw) / max(1.0, float(bh))
    tip_travel = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
    closure_gap = tip_travel / max(1.0, diag)

    is_loop_candidate = (
        min(bh, bw) >= max(6, int(min_dim * 0.032))
        and 0.24 <= aspect <= 4.2
        and fill_density <= 0.46
        and (
            hole_area >= 0.10 * bbox_area
            or (angular_coverage_deg >= 270 and closure_gap <= 0.38)
            or (angular_coverage_deg >= 195 and ellipse_cv <= 0.105 and fill_density <= 0.36)
        )
    )

    if is_loop_candidate:
        is_closed = bool(
            hole_area >= 0.10 * bbox_area or (angular_coverage_deg >= 300 and closure_gap <= 0.35)
        )
        shape_type = (
            "circle_draw"
            if (ellipse_cv <= 0.105 and 0.38 <= aspect <= 2.65 and angular_coverage_deg >= 195)
            else "contour_outline_draw"
        )

        uniq_t = np.unique(birth_frames)
        tip_angles = []
        for tb in uniq_t:
            m_t = birth_frames == tb
            if int(np.sum(m_t)) >= 1:
                tip_angles.append(float(np.arctan2(np.mean(dy[m_t]), np.mean(dx[m_t]))))
        if len(tip_angles) >= 5:
            unwrapped = np.unwrap(np.asarray(tip_angles))
            ang_rho = _spearman_corr(unwrapped)
            if ang_rho >= 0.65:
                sweep_dir = "clockwise"
            elif ang_rho <= -0.65:
                sweep_dir = "counter_clockwise"
            else:
                sweep_dir = "symmetric_bilateral"
        else:
            sweep_dir = "clockwise"

        return {
            "shape_type": shape_type,
            "is_closed_loop": is_closed,
            "sweep_direction": sweep_dir,
            "angular_coverage_deg": angular_coverage_deg,
            "ellipse_cv": round(ellipse_cv, 4),
            "fill_density": round(fill_density, 4),
            "stroke_radius_px": round(mean_radius, 3),
            "center_norm": [round(cx / work_w, 4), round(cy / work_h, 4)],
            "start_point_norm": [round(start_xy[0] / work_w, 4), round(start_xy[1] / work_h, 4)],
            "end_point_norm": [round(end_xy[0] / work_w, 4), round(end_xy[1] / work_h, 4)],
        }

    # Check progressive Callout / Pointer / Indicator Line Draw!
    if (
        diag >= min_dim * 0.13
        and mean_radius <= 2.05
        and fill_density <= 0.48
        and closure_gap >= 0.45
    ):
        return {
            "shape_type": "callout_line_draw",
            "is_closed_loop": False,
            "sweep_direction": "linear_extend",
            "angular_coverage_deg": angular_coverage_deg,
            "ellipse_cv": round(ellipse_cv, 4),
            "fill_density": round(fill_density, 4),
            "stroke_radius_px": round(mean_radius, 3),
            "center_norm": [round(cx / work_w, 4), round(cy / work_h, 4)],
            "start_point_norm": [round(start_xy[0] / work_w, 4), round(start_xy[1] / work_h, 4)],
            "end_point_norm": [round(end_xy[0] / work_w, 4), round(end_xy[1] / work_h, 4)],
        }

    return None


def detect_frames_stroke_draws(
    frames: List[np.ndarray],
    fps: float = 30.0,
    min_duration_s: float = 0.28,
    allowed_shapes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detects progressive circle draws, object contour outlines, and callout/pointer line draws in RGB frames."""
    n_frames = len(frames)
    if n_frames == 0:
        return {"n_frames": 0, "fps": fps, "stroke_draw_count": 0, "stroke_draws": []}

    orig_h, orig_w = frames[0].shape[:2]
    if orig_h >= orig_w:
        work_h, work_w = 360, 202
    else:
        work_h, work_w = 202, 360

    rgb_frames = np.zeros((n_frames, work_h, work_w, 3), dtype=np.float32)
    luma_frames = np.zeros((n_frames, work_h, work_w), dtype=np.float32)
    sat_frames = np.zeros((n_frames, work_h, work_w), dtype=np.float32)

    for i, f in enumerate(frames):
        rgb = resize_float(f, work_h, work_w, anti_aliasing=True).astype(np.float32)
        rgb_frames[i] = rgb
        luma_frames[i] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        sat_frames[i] = rgb2hsv_fast(rgb)[:, :, 1]

    # Segment stationary camera shots + overlapping sub-windows (preserving parent shot bounds!)
    frame_diff = np.mean(np.abs(np.diff(luma_frames, axis=0)), axis=(1, 2))
    cuts = [0]
    anchor = luma_frames[0]
    for i in range(1, n_frames):
        g_shift = float(np.mean(np.abs(luma_frames[i] - anchor) > 0.14))
        if float(frame_diff[i - 1]) > 0.075 or (g_shift > 0.27 and (i - cuts[-1]) >= int(fps * 0.7)):
            cuts.append(i)
            anchor = luma_frames[i]
    cuts.append(n_frames)

    min_frames = max(7, int(round(min_duration_s * fps)))
    scan_windows: List[Tuple[int, int, int]] = []  # (ws, we, parent_shot_end)
    for k in range(len(cuts) - 1):
        s0, s1 = cuts[k], cuts[k + 1]
        if (s1 - s0) < min_frames:
            continue
        scan_windows.append((s0, s1, s1))
        for sub_s in (3.4, 1.50, 0.85):
            w_len = int(round(sub_s * fps))
            w_stride = max(min_frames // 2, w_len // 2)
            if (s1 - s0) > int(w_len * 1.15) and w_len >= min_frames:
                for ws in range(s0, s1 - min_frames + 1, w_stride):
                    we = min(s1, ws + w_len)
                    if (we - ws) >= min_frames:
                        scan_windows.append((ws, we, s1))

    allowed_set = (
        set(allowed_shapes)
        if allowed_shapes
        else {"circle_draw", "contour_outline_draw", "callout_line_draw"}
    )
    candidates: List[Dict[str, Any]] = []
    hud_top_y = int(work_h * 0.15)

    for s0, s1, parent_end in scan_windows:
        t_len = s1 - s0
        seg_rgb = rgb_frames[s0:s1]
        seg_luma = luma_frames[s0:s1]
        seg_sat = sat_frames[s0:s1]

        base_rgb = np.median(seg_rgb[: min(3, t_len)], axis=0)
        base_luma = np.median(seg_luma[: min(3, t_len)], axis=0)
        base_sat = np.median(seg_sat[: min(3, t_len)], axis=0)

        peak_rgb_diff = np.percentile(
            np.max(np.abs(seg_rgb - base_rgb[None, ...]), axis=-1), 96, axis=0
        )
        peak_luma_gain = np.percentile(seg_luma, 96, axis=0) - base_luma
        peak_sat_gain = np.percentile(seg_sat, 96, axis=0) - base_sat
        peak_luma = np.percentile(seg_luma, 96, axis=0)

        # Added-ink proposal: catches BOTH bright white strokes AND colored contours over midtone organs/CTs!
        added_mask = (peak_rgb_diff > 0.15) & (
            (peak_luma_gain > 0.09) | ((peak_sat_gain > 0.14) & (peak_luma > 0.18))
        )
        labeled, n_comp = label(binary_dilation(added_mask, iterations=1))

        for cid in range(1, n_comp + 1):
            ys, xs = np.where((labeled == cid) & added_mask)
            area = len(ys)
            if area < 15 or area > int(work_h * work_w * 0.10):
                continue

            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            if x0 <= 1 or y0 <= 1 or x1 >= work_w - 1 or y1 >= work_h - 1:
                continue
            bh, bw = y1 - y0, x1 - x0
            if max(bh, bw) < 9:
                continue

            pix_rgb_diff = np.max(
                np.abs(seg_rgb[:, ys, xs, :] - base_rgb[ys, xs, :][None, :, :]), axis=-1
            )
            pix_luma_gain = seg_luma[:, ys, xs] - base_luma[ys, xs][None, :]
            pix_sat_gain = seg_sat[:, ys, xs] - base_sat[ys, xs][None, :]
            on_mat = (pix_rgb_diff > 0.13) & (
                (pix_luma_gain > 0.08)
                | ((pix_sat_gain > 0.13) & (seg_luma[:, ys, xs] > 0.18))
            )

            counts = np.sum(on_mat, axis=1)
            peak_count = int(np.max(counts))
            if peak_count < 14 or peak_count < area * 0.48:
                continue

            reach_98 = np.where(counts >= max(1, int(peak_count * 0.985)))[0]
            t_hold_end = int(np.argmax(counts))
            t_finish_cand = int(reach_98[0]) if len(reach_98) > 0 else t_hold_end
            if t_finish_cand < min_frames - 1:
                continue

            valid_pix = on_mat[t_finish_cand]
            n_valid = int(np.sum(valid_pix))
            if n_valid < 14:
                continue

            # POST-FINISH PERSISTENCE CHECK IN PARENT SHOT:
            # Rejects sweeping scan lines that pass through a sub-window right at its end!
            abs_finish_idx = s0 + t_finish_cand
            post_check_idx = min(parent_end - 1, abs_finish_idx + 5)
            if post_check_idx > abs_finish_idx:
                v_ys_chk, v_xs_chk = ys[valid_pix], xs[valid_pix]
                post_rgb_d = np.max(
                    np.abs(rgb_frames[post_check_idx, v_ys_chk, v_xs_chk, :] - base_rgb[v_ys_chk, v_xs_chk, :]),
                    axis=-1,
                )
                if float(np.mean(post_rgb_d > 0.12)) < 0.72:
                    continue

            on_sub = on_mat[: t_finish_cand + 1, valid_pix]
            first_on = np.argmax(on_sub, axis=0)
            growing_idx = np.where(first_on > 0)[0]
            if len(growing_idx) < 12:
                continue

            pers_scores = [float(np.mean(on_sub[int(first_on[j]) :, j])) for j in growing_idx]
            mean_pers = float(np.mean(pers_scores))
            if mean_pers < 0.80:
                continue

            cum_counts = np.sum(on_sub, axis=1).astype(np.float64)
            c_smooth = uniform_filter1d(cum_counts, size=2)
            c_min, c_max = float(np.min(c_smooth)), float(np.max(c_smooth))
            c_span = c_max - c_min
            if c_span < 12:
                continue

            above_start = np.where(c_smooth <= c_min + 0.04 * c_span)[0]
            t0 = int(above_start[-1]) if len(above_start) > 0 else 0
            above_end = np.where(c_smooth >= c_min + 0.98 * c_span)[0]
            t1 = (int(above_end[0]) + 1) if len(above_end) > 0 else (t_finish_cand + 1)
            t1 = min(t_finish_cand + 1, max(t0 + 2, t1))
            m_len = t1 - t0
            if m_len < min_frames:
                continue

            # STATIONARY-BACKGROUND CHECK: Rejects global camera zooms, pans, and cross-dissolves!
            bg_mask = np.ones((work_h, work_w), dtype=bool)
            bg_mask[:hud_top_y, :] = False
            bg_mask[max(0, y0 - 6) : min(work_h, y1 + 6), max(0, x0 - 6) : min(work_w, x1 + 6)] = False
            if int(np.sum(bg_mask)) > 100:
                bg_diff = np.abs(seg_luma[t1 - 1, bg_mask] - seg_luma[t0, bg_mask])
                if float(np.mean(bg_diff > 0.09)) > 0.095:
                    continue

            # LOCAL NET INK ADDITION CHECK: Rejects translating/drifting molecules & objects!
            pad = 6
            py0, py1 = max(0, y0 - pad), min(work_h, y1 + pad)
            px0, px1 = max(0, x0 - pad), min(work_w, x1 + pad)
            rgb_d01 = np.max(
                np.abs(seg_rgb[t1 - 1, py0:py1, px0:px1] - seg_rgb[t0, py0:py1, px0:px1]),
                axis=-1,
            )
            luma_g01 = seg_luma[t1 - 1, py0:py1, px0:px1] - seg_luma[t0, py0:py1, px0:px1]
            sat_g01 = seg_sat[t1 - 1, py0:py1, px0:px1] - seg_sat[t0, py0:py1, px0:px1]
            n_added_box = int(
                np.sum(
                    (rgb_d01 > 0.14)
                    & ((luma_g01 > 0.08) | ((sat_g01 > 0.13) & (seg_luma[t1 - 1, py0:py1, px0:px1] > 0.18)))
                )
            )
            n_vanished_box = int(
                np.sum((rgb_d01 > 0.14) & ((luma_g01 < -0.09) | (sat_g01 < -0.15)))
            )
            if n_added_box < 12:
                continue
            net_add_ratio = float(n_added_box - n_vanished_box) / float(n_added_box)
            if net_add_ratio < 0.68:
                continue

            raw_sub = c_smooth[t0:t1]
            draw_curve = np.clip(
                (raw_sub - raw_sub[0]) / max(1e-5, raw_sub[-1] - raw_sub[0]), 0.0, 1.0
            )
            rho = _spearman_corr(draw_curve)
            if rho < 0.90:
                continue

            inter_ratio = float(np.mean((draw_curve >= 0.14) & (draw_curve <= 0.88)))
            max_step = float(np.max(np.abs(np.diff(draw_curve))))
            if inter_ratio < 0.25 or max_step > 0.38:
                continue

            v_ys = ys[valid_pix]
            v_xs = xs[valid_pix]
            v_birth = np.clip(first_on, t0, t1 - 1) - t0

            # FADE-IN REJECTION CHECK
            mid_35_idx = int(np.argmin(np.abs(draw_curve - 0.35)))
            early_mask = v_birth <= max(1, mid_35_idx)
            if int(np.sum(early_mask)) >= 4:
                early_span = float(np.hypot(np.ptp(v_xs[early_mask]), np.ptp(v_ys[early_mask])))
                full_span = float(np.hypot(bw, bh))
                cy_all, cx_all = float(np.mean(v_ys)), float(np.mean(v_xs))
                sec_all = len(
                    np.unique(
                        np.floor((np.arctan2(v_ys - cy_all, v_xs - cx_all) + np.pi) / (2 * np.pi) * 24).astype(int) % 24
                    )
                )
                sec_early = len(
                    np.unique(
                        np.floor(
                            (np.arctan2(v_ys[early_mask] - cy_all, v_xs[early_mask] - cx_all) + np.pi)
                            / (2 * np.pi)
                            * 24
                        ).astype(int)
                        % 24
                    )
                )
                if early_span > 0.88 * full_span and sec_early > 0.82 * max(1, sec_all):
                    continue

            # Moving pen-tip front continuity check
            cont_hits = []
            for k_step in range(1, m_len):
                new_k = v_birth == k_step
                prev_k = v_birth < k_step
                if int(np.sum(new_k)) == 0 or int(np.sum(prev_k)) == 0:
                    continue
                ny, nx = v_ys[new_k][:, None], v_xs[new_k][:, None]
                py_p, px_p = v_ys[prev_k][None, :], v_xs[prev_k][None, :]
                min_d = np.min(np.sqrt((ny - py_p) ** 2 + (nx - px_p) ** 2), axis=1)
                cont_hits.append(float(np.mean(min_d <= 5.2)))
            front_continuity = float(np.mean(cont_hits)) if cont_hits else 0.0
            if front_continuity < 0.62:
                continue

            geom = _classify_stroke_geometry(
                v_ys, v_xs, v_birth, seg_rgb[t1 - 1], work_h, work_w
            )
            if geom is None or geom["shape_type"] not in allowed_set:
                continue

            abs_s = s0 + t0
            abs_e = s0 + t1
            dur_s = (abs_e - abs_s) / fps
            if dur_s < min_duration_s:
                continue

            finish_s = round(abs_e / fps, 3)
            is_loop = geom["shape_type"] in ("circle_draw", "contour_outline_draw")
            cov_factor = 0.72 + 0.28 * min(1.0, geom["angular_coverage_deg"] / 345.0)
            span_factor = min(1.25, max(0.85, dur_s / 0.55))
            quality = round(
                rho
                * mean_pers
                * net_add_ratio
                * front_continuity
                * cov_factor
                * span_factor
                * (1.35 if geom["shape_type"] == "circle_draw" else (1.20 if is_loop else 1.0)),
                4,
            )

            candidates.append(
                {
                    "start_frame": int(abs_s),
                    "end_frame": int(abs_e),
                    "finish_frame": int(abs_e),
                    "start_s": round(abs_s / fps, 3),
                    "end_s": finish_s,
                    "finish_s": finish_s,
                    "duration_s": round(dur_s, 3),
                    "shape_type": geom["shape_type"],
                    "is_closed_loop": geom["is_closed_loop"],
                    "sweep_direction": geom["sweep_direction"],
                    "bbox_norm": [
                        round(x0 / work_w, 4),
                        round(y0 / work_h, 4),
                        round(x1 / work_w, 4),
                        round(y1 / work_h, 4),
                    ],
                    "center_norm": geom["center_norm"],
                    "start_point_norm": geom["start_point_norm"],
                    "end_point_norm": geom["end_point_norm"],
                    "angular_coverage_deg": geom["angular_coverage_deg"],
                    "stroke_radius_px": geom["stroke_radius_px"],
                    "monotonicity": round(rho, 4),
                    "persistence": round(mean_pers, 4),
                    "net_addition_ratio": round(net_add_ratio, 4),
                    "front_continuity": round(front_continuity, 4),
                    "quality_score": quality,
                    "song_part_mapping": (
                        {
                            "section_type": "circular_loop_trace_part",
                            "arpeggio_orbit_follows_draw_curve": True,
                            "loop_closure_chime_at_finish_s": finish_s,
                            "recommended_preset": "pc98_opna_glass_arp",
                            "finish_accent_preset": "trig_confirm_chime",
                        }
                        if is_loop
                        else {
                            "section_type": "indicator_line_glide_part",
                            "pitch_glide_follows_draw_curve": True,
                            "target_lock_ping_at_finish_s": finish_s,
                            "recommended_preset": "aloboi_nostalgic_pluck",
                            "finish_accent_preset": "trig_micro_droplet_ping",
                        }
                    ),
                    "draw_curve": [round(float(v), 4) for v in draw_curve],
                }
            )

    def _bbox_iou(b1: List[float], b2: List[float]) -> float:
        ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter / max(1e-6, a1 + a2 - inter)

    candidates.sort(key=lambda d: (-d["quality_score"], d["start_s"]))
    merged: List[Dict[str, Any]] = []
    for cand in candidates:
        dup = False
        for ex in merged:
            t_ov = min(cand["end_s"], ex["end_s"]) - max(cand["start_s"], ex["start_s"])
            if (
                t_ov > 0.25 * min(cand["duration_s"], ex["duration_s"])
                and _bbox_iou(cand["bbox_norm"], ex["bbox_norm"]) > 0.18
            ):
                dup = True
                break
        if not dup:
            merged.append(cand)

    merged.sort(key=lambda d: d["start_s"])
    by_type: Dict[str, int] = {}
    for d in merged:
        by_type[d["shape_type"]] = by_type.get(d["shape_type"], 0) + 1

    return {
        "schema": "pyxelate-detected-stroke-draws-v1",
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": round(n_frames / fps, 3),
        "stroke_draw_count": len(merged),
        "counts_by_shape": by_type,
        "stroke_draws": merged,
    }


def detect_video_stroke_draws(
    video_path: str,
    max_frames: Optional[int] = None,
    min_duration_s: float = 0.28,
    allowed_shapes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reads a video file from disk and detects progressive circle, contour outline, and callout line draws."""
    fps = 30.0
    try:
        fps = float(iio.immeta(video_path).get("fps", 30.0))
    except Exception:
        fps = 30.0

    frames = []
    for frame in iio.imiter(video_path):
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    res = detect_frames_stroke_draws(
        frames,
        fps=fps,
        min_duration_s=min_duration_s,
        allowed_shapes=allowed_shapes,
    )
    res["video_path"] = video_path
    return res


def render_stroke_draws_proof(
    frames: List[np.ndarray],
    result: Dict[str, Any],
    output_image_path: str,
    max_rows: int = 10,
) -> Optional[str]:
    """Renders a visual verification contact sheet (`START -> MID DRAWING -> FINISH + draw_curve`) for detected draws."""
    from PIL import Image, ImageDraw

    draws = result.get("stroke_draws", [])[:max_rows]
    if not draws or not frames:
        return None

    fps = float(result.get("fps", 30.0))
    oh, ow = frames[0].shape[:2]
    if oh >= ow:
        tw, th = 190, 338
    else:
        tw, th = 300, 170
    chart_w = 270
    canvas = Image.new("RGB", (tw * 3 + chart_w, th * len(draws)), (10, 12, 22))
    cdraw = ImageDraw.Draw(canvas)

    palette_map = {
        "circle_draw": (40, 250, 195),
        "contour_outline_draw": (255, 195, 50),
        "callout_line_draw": (115, 185, 255),
    }

    for r_i, d in enumerate(draws):
        y_off = r_i * th
        sf = max(0, min(len(frames) - 1, int(d["start_frame"])))
        ef = max(0, min(len(frames) - 1, int(d["finish_frame"]) - 1))
        mf = (sf + ef) // 2
        bx0, by0, bx1, by1 = d["bbox_norm"]
        col = palette_map.get(d["shape_type"], (0, 255, 180))

        for c_i, (lbl, f_idx) in enumerate([("START", sf), ("DRAWING", mf), ("FINISH", ef)]):
            im = Image.fromarray(frames[f_idx]).resize((tw, th))
            idraw = ImageDraw.Draw(im)
            rx0, ry0 = max(0, int(bx0 * tw) - 3), max(0, int(by0 * th) - 3)
            rx1, ry1 = min(tw - 1, int(bx1 * tw) + 3), min(th - 1, int(by1 * th) + 3)
            idraw.rectangle([rx0, ry0, rx1, ry1], outline=col, width=2)
            idraw.rectangle([0, 0, tw - 1, 20], fill=(0, 0, 0))
            idraw.text((4, 4), f"{lbl} {f_idx / fps:.2f}s", fill=col)
            canvas.paste(im, (c_i * tw, y_off))

        cx0 = tw * 3 + 12
        cy0 = y_off + 28
        cw = chart_w - 24
        ch = th - 58
        cdraw.rectangle([cx0, cy0, cx0 + cw, cy0 + ch], fill=(18, 22, 38), outline=(72, 82, 130))
        cdraw.text((cx0, y_off + 6), f"#{r_i + 1} {d['shape_type']} ({d['sweep_direction']})", fill=col)
        curve = d.get("draw_curve", [])
        if len(curve) > 1:
            pts = [
                (
                    cx0 + int(round((k / (len(curve) - 1)) * cw)),
                    cy0 + ch - int(round(curve[k] * (ch - 8))) - 4,
                )
                for k in range(len(curve))
            ]
            cdraw.line(pts, fill=col, width=3)
        cdraw.text(
            (cx0, cy0 + ch + 6),
            f"{d['start_s']:.2f}s -> finish {d['finish_s']:.2f}s | ang={d['angular_coverage_deg']}deg",
            fill=(205, 215, 242),
        )

    out_p = Path(output_image_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_p))
    return str(out_p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect progressive circle, contour outline, and callout line drawings in a video."
    )
    parser.add_argument("video_path", help="Path to input video file.")
    parser.add_argument("--min-duration", type=float, default=0.28, help="Minimum draw duration in seconds.")
    parser.add_argument(
        "--shapes",
        type=str,
        default="circle_draw,contour_outline_draw,callout_line_draw",
        help="Comma-separated shape types to include.",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON file path.")
    parser.add_argument("--proof", type=str, default=None, help="Optional output visual proof image path.")
    args = parser.parse_args()

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    fps = 30.0
    try:
        fps = float(iio.immeta(args.video_path).get("fps", 30.0))
    except Exception:
        fps = 30.0
    frames = [f for f in iio.imiter(args.video_path)]
    res = detect_frames_stroke_draws(frames, fps=fps, min_duration_s=args.min_duration, allowed_shapes=shapes)
    res["video_path"] = args.video_path

    print(
        f"Detected {res['stroke_draw_count']} progressive indicator draws "
        f"(by shape: {res['counts_by_shape']}):"
    )
    for idx, d in enumerate(res["stroke_draws"], 1):
        print(
            f"  #{idx:2d}: [{d['start_s']:6.2f}s -> finish {d['finish_s']:6.2f}s] ({d['duration_s']:.2f}s) "
            f"{d['shape_type']:21s} | {d['sweep_direction']:19s} | closed={str(d['is_closed_loop']):5s} "
            f"| bbox={d['bbox_norm']} | mono={d['monotonicity']}"
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved JSON detections: {args.json}")

    if args.proof:
        saved = render_stroke_draws_proof(frames, res, args.proof)
        if saved:
            print(f"Saved visual proof sheet: {saved}")


if __name__ == "__main__":
    main()
