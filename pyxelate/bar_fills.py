"""Computer-Vision Horizontal & Vertical Bar-Fill Detector for Exact Song-Part Synchronization.

Detects 1D horizontal (`horizontal_ltr`, `horizontal_rtl`) and vertical (`vertical_btt`, `vertical_ttb`)
progress bars, energy/dose meters, gauges, liquid/beam column fills, and stacked multi-bar arrays
(such as multileaf collimators or bar charts) in any video, returning frame-accurate `fill_curve`
trajectories in [0.0, 1.0] so a synthesizer/sequencer can match each bar fill 1:1 as a dedicated
musical song section.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from scipy.ndimage import uniform_filter1d, label
import imageio.v3 as iio

from .fast_ops import resize_float, rgb2hsv_fast


def _monotonic_spearman(arr: np.ndarray) -> float:
    """Fast Spearman rank correlation of `arr` against time index `0..N-1` in [-1, 1]."""
    n = len(arr)
    if n < 5:
        return 0.0
    if float(np.std(arr)) < 1e-6:
        return 0.0
    ranks = np.argsort(np.argsort(arr)).astype(np.float64)
    t = np.arange(n, dtype=np.float64)
    r_c = ranks - np.mean(ranks)
    t_c = t - np.mean(t)
    denom = float(np.sqrt(np.sum(r_c * r_c) * np.sum(t_c * t_c)))
    return float(np.sum(r_c * t_c) / denom) if denom > 1e-9 else 0.0


def _evaluate_1d_bar_profile(
    profile_2d: np.ndarray, axis_name: str, allow_symmetric_inward: bool = False
) -> Optional[Dict[str, Any]]:
    """Evaluates a `(T, L)` 1D spatial-temporal strip profile for progressive bar filling or draining.

    Rejects instant step-jumps (e.g. objects popping in/out or passing across a region) by requiring:
    - Trimming post-100% UI fade-outs cleanly at the completion extremum.
    - High linear/progressive ramp R^2 (>= 0.78) and Spearman monotonicity (|rho| >= 0.84).
    - Genuine intermediate traversal (>= 28% of frames in [0.15, 0.85]) and max single-frame step < 0.36.
    """
    t_len, s_len = profile_2d.shape
    if t_len < 12 or s_len < 7:
        return None

    p_min = np.percentile(profile_2d, 8, axis=0, keepdims=True)
    p_max = float(np.percentile(profile_2d, 97))
    dyn_range = p_max - float(np.mean(p_min))
    if dyn_range < 0.075:
        return None

    norm = np.clip((profile_2d - p_min) / max(1e-5, dyn_range), 0.0, 1.0)
    active = norm > 0.36  # (T, L)

    frac_raw = np.mean(active, axis=1).astype(np.float64)
    frac_smooth = uniform_filter1d(frac_raw, size=3)
    span = float(np.max(frac_smooth) - np.min(frac_smooth))
    if span < 0.30:
        return None

    # Evaluate both the primary pre-peak FILL interval and post-peak DRAIN interval so that
    # an instant UI widget disappearance after 100% completion never masks a genuine multi-second fill!
    idx_max = int(np.argmax(frac_smooth))
    idx_min = int(np.argmin(frac_smooth))
    candidate_windows = []
    if idx_max >= 9:
        pre_min_idx = int(np.argmin(frac_smooth[: idx_max + 1]))
        if idx_max - pre_min_idx >= 9:
            candidate_windows.append((pre_min_idx, idx_max))
    if (t_len - 1 - idx_max) >= 9:
        post_min_idx = idx_max + int(np.argmin(frac_smooth[idx_max:]))
        if post_min_idx - idx_max >= 9:
            candidate_windows.append((idx_max, post_min_idx))
    if idx_min >= 9:
        pre_max_idx = int(np.argmax(frac_smooth[: idx_min + 1]))
        if idx_min - pre_max_idx >= 9 and (pre_max_idx, idx_min) not in candidate_windows:
            candidate_windows.append((pre_max_idx, idx_min))

    best_result: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for w_start, w_end in candidate_windows:
        window_frac = frac_smooth[w_start : w_end + 1]
        w_span = float(np.max(window_frac) - np.min(window_frac))
        if len(window_frac) < 10 or w_span < 0.30:
            continue

        # Trim leading/trailing flat plateaus inside [w_start..w_end]
        f_lo = float(np.min(window_frac)) + 0.05 * w_span
        f_hi = float(np.max(window_frac)) - 0.04 * w_span
        changing = np.where((window_frac >= f_lo) & (window_frac <= f_hi))[0]
        if len(changing) < 7:
            continue

        loc0 = max(0, int(changing[0]) - 1)
        loc1 = min(len(window_frac), int(changing[-1]) + 2)
        t0 = w_start + loc0
        t1 = w_start + loc1
        m = t1 - t0
        if m < 10:
            continue

        sub_act = active[t0:t1]
        sub_frac = frac_smooth[t0:t1]
        c_min, c_max = float(np.min(sub_frac)), float(np.max(sub_frac))
        if (c_max - c_min) < 0.28:
            continue

        curve = np.clip((sub_frac - c_min) / (c_max - c_min), 0.0, 1.0)
        rho = _monotonic_spearman(curve)
        if abs(rho) < 0.85:
            continue

        is_filling = rho > 0
        ideal_ramp = np.linspace(0.0, 1.0, m) if is_filling else np.linspace(1.0, 0.0, m)
        # Allow smooth accelerating/decelerating ramps (power 0.65..1.6) as well as linear ramps
        best_r2 = -1.0
        for gamma in (0.65, 1.0, 1.5):
            ref = ideal_ramp ** gamma if is_filling else 1.0 - (1.0 - ideal_ramp) ** gamma
            ss_res = float(np.sum((curve - ref) ** 2))
            ss_tot = float(np.sum((curve - np.mean(curve)) ** 2)) + 1e-9
            best_r2 = max(best_r2, 1.0 - ss_res / ss_tot)
        if best_r2 < 0.78:
            continue

        # Reject instant pop-in / step-jump false positives!
        intermediate_ratio = float(np.mean((curve >= 0.15) & (curve <= 0.85)))
        max_step = float(np.max(np.abs(np.diff(curve))))
        if intermediate_ratio < 0.28 or max_step > 0.36:
            continue

        # Spatial fill-front coherence check along active columns/rows of bar axis `0..L-1`
        col_changing = np.max(sub_act, axis=0) != np.min(sub_act, axis=0)
        act_cols = sub_act[:, col_changing] if int(np.sum(col_changing)) >= 6 else sub_act
        eff_len = act_cols.shape[1]

        fwd_scores = []
        rev_scores = []
        sym_scores = []
        for row in act_cols:
            k = int(round(np.sum(row)))
            if 1 <= k <= eff_len - 1:
                fwd_scores.append(float(np.mean(row[:k])) - float(np.mean(row[k:])))
                rev_scores.append(float(np.mean(row[eff_len - k :])) - float(np.mean(row[: eff_len - k])))
                if allow_symmetric_inward and k >= 2:
                    half_l = k // 2
                    half_r = k - half_l
                    edges_mean = float(np.sum(row[:half_l]) + np.sum(row[eff_len - half_r :])) / k
                    mid_mean = float(np.mean(row[half_l : eff_len - half_r]))
                    sym_scores.append(edges_mean - mid_mean)

        if not fwd_scores:
            continue
        fwd_coh = float(np.mean(fwd_scores))
        rev_coh = float(np.mean(rev_scores))
        sym_coh = float(np.mean(sym_scores)) if sym_scores else -1.0
        best_coh = max(fwd_coh, rev_coh, sym_coh)
        if best_coh < 0.48:
            continue

        if sym_coh >= max(fwd_coh, rev_coh) and allow_symmetric_inward:
            direction = "horizontal_inward" if axis_name == "horizontal" else "vertical_inward"
        else:
            is_forward = fwd_coh >= rev_coh
            if axis_name == "horizontal":
                direction = "horizontal_ltr" if is_forward else "horizontal_rtl"
            else:
                direction = "vertical_ttb" if is_forward else "vertical_btt"

        cand_score = float(m) * w_span * abs(rho) * best_r2 * best_coh
        if cand_score > best_score:
            best_score = cand_score
            best_result = {
                "local_start": int(t0),
                "local_end": int(t1),
                "orientation": axis_name,
                "direction": direction,
                "action": "fill" if is_filling else "drain",
                "monotonicity": round(abs(rho), 4),
                "ramp_r2": round(best_r2, 4),
                "coherence": round(best_coh, 4),
                "intermediate_ratio": round(intermediate_ratio, 4),
                "fill_curve": [round(float(v), 4) for v in curve],
            }

    return best_result


def detect_frames_bar_fills(
    frames: List[np.ndarray],
    fps: float = 30.0,
    min_duration_s: float = 0.65,
) -> Dict[str, Any]:
    """Detects horizontal and vertical bar fills/drains (single bars & stacked bar arrays) in RGB frames."""
    n_frames = len(frames)
    if n_frames == 0:
        return {"n_frames": 0, "fps": fps, "bar_fill_count": 0, "bar_fills": []}

    orig_h, orig_w = frames[0].shape[:2]
    if orig_h >= orig_w:
        work_h, work_w = 256, 144
    else:
        work_h, work_w = 144, 256

    sig_frames = np.zeros((n_frames, work_h, work_w), dtype=np.float32)
    luma_frames = np.zeros((n_frames, work_h, work_w), dtype=np.float32)

    for i, f in enumerate(frames):
        rgb = resize_float(f, work_h, work_w, anti_aliasing=True)
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        sat = rgb2hsv_fast(rgb)[:, :, 1]
        luma_frames[i] = luma.astype(np.float32)
        sig_frames[i] = np.clip(
            luma * 0.65 + sat * np.maximum(luma, 0.25) * 0.85, 0.0, 1.2
        ).astype(np.float32)

    # Segment into stationary-camera windows + multi-scale sub-windows (2.2s and 4.5s)
    # so short bar fills inside long stationary shots are never diluted!
    frame_diff = np.mean(np.abs(np.diff(luma_frames, axis=0)), axis=(1, 2))
    cut_indices = [0]
    seg_anchor = luma_frames[0]
    for i in range(1, n_frames):
        global_shift = float(np.mean(np.abs(luma_frames[i] - seg_anchor) > 0.14))
        instant_cut = float(frame_diff[i - 1]) > 0.078
        if instant_cut or (global_shift > 0.28 and (i - cut_indices[-1]) >= int(fps * 0.8)):
            cut_indices.append(i)
            seg_anchor = luma_frames[i]
    cut_indices.append(n_frames)

    min_frames = max(10, int(round(min_duration_s * fps)))
    scan_windows = []
    for seg_i in range(len(cut_indices) - 1):
        s0, s1 = cut_indices[seg_i], cut_indices[seg_i + 1]
        if (s1 - s0) < min_frames:
            continue
        scan_windows.append((s0, s1))
        for sub_sec in (4.5, 2.2):
            w_len = int(round(sub_sec * fps))
            w_stride = max(min_frames // 2, w_len // 2)
            if (s1 - s0) > int(w_len * 1.25):
                for ws in range(s0, s1 - min_frames + 1, w_stride):
                    we = min(s1, ws + w_len)
                    if (we - ws) >= min_frames:
                        scan_windows.append((ws, we))

    detected: List[Dict[str, Any]] = []

    for s0, s1 in scan_windows:
        seg_sig = sig_frames[s0:s1]
        seg_luma = luma_frames[s0:s1]
        t_range = np.percentile(seg_sig, 97, axis=0) - np.percentile(seg_sig, 5, axis=0)
        thresh = max(0.11, min(0.28, float(np.percentile(t_range, 91))))
        active_mask = t_range >= thresh

        labeled, n_comp = label(active_mask)
        for comp_id in range(1, n_comp + 1):
            ys, xs = np.where(labeled == comp_id)
            if len(ys) < 10:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            # Reject 1-pixel screen-edge clipping artifacts (objects drifting off frame edge)
            if x0 <= 1 or y0 <= 1 or x1 >= work_w - 1 or y1 >= work_h - 1:
                continue
            bh, bw = y1 - y0, x1 - x0
            aspect = bw / max(1.0, float(bh))

            is_horiz = aspect >= 3.2 and bw >= int(work_w * 0.13) and bh <= int(work_h * 0.16)
            is_vert = (1.0 / max(1e-4, aspect)) >= 3.2 and bh >= int(work_h * 0.12) and bw <= int(work_w * 0.16)

            # Check for stacked multi-bar arrays (e.g. 20 horizontal collimator leaves or vertical equalizer bars!)
            is_stacked_horiz = False
            is_stacked_vert = False
            if not (is_horiz or is_vert) and bw >= int(work_w * 0.22) and bh >= int(work_h * 0.15):
                mid_frame = seg_luma[len(seg_luma) // 2, y0:y1, x0:x1]
                gy = float(np.mean(np.abs(np.diff(mid_frame, axis=0))))
                gx = float(np.mean(np.abs(np.diff(mid_frame, axis=1))))
                if gy > gx * 1.55 and gy > 0.035:
                    is_stacked_horiz = True
                elif gx > gy * 1.55 and gx > 0.035:
                    is_stacked_vert = True

            if not (is_horiz or is_vert or is_stacked_horiz or is_stacked_vert):
                continue

            use_horiz = is_horiz or is_stacked_horiz
            # Interior-to-Exterior Strip Contrast Check
            interior_energy = float(np.mean(t_range[y0:y1, x0:x1]))
            if use_horiz:
                pad = max(2, min(12, bh))
                top_ext = t_range[max(0, y0 - pad) : y0, x0:x1]
                bot_ext = t_range[y1 : min(work_h, y1 + pad), x0:x1]
                ext_vals = np.concatenate([top_ext.ravel(), bot_ext.ravel()])
            else:
                pad = max(2, min(12, bw))
                left_ext = t_range[y0:y1, max(0, x0 - pad) : x0]
                right_ext = t_range[y0:y1, x1 : min(work_w, x1 + pad)]
                ext_vals = np.concatenate([left_ext.ravel(), right_ext.ravel()])

            exterior_energy = float(np.mean(ext_vals)) if len(ext_vals) > 0 else 0.0
            contrast_ratio = interior_energy / max(0.015, exterior_energy)
            if contrast_ratio < 2.1:
                continue

            if use_horiz:
                prof = np.mean(seg_sig[:, y0:y1, x0:x1], axis=1)
                eval_res = _evaluate_1d_bar_profile(
                    prof, "horizontal", allow_symmetric_inward=is_stacked_horiz
                )
            else:
                prof = np.mean(seg_sig[:, y0:y1, x0:x1], axis=2)
                eval_res = _evaluate_1d_bar_profile(
                    prof, "vertical", allow_symmetric_inward=is_stacked_vert
                )

            if eval_res is None:
                continue

            abs_s = s0 + eval_res["local_start"]
            abs_e = s0 + eval_res["local_end"]
            dur = (abs_e - abs_s) / fps
            if dur < min_duration_s:
                continue

            bbox_norm = [
                round(x0 / work_w, 4),
                round(y0 / work_h, 4),
                round(x1 / work_w, 4),
                round(y1 / work_h, 4),
            ]
            quality = round(
                eval_res["monotonicity"]
                * eval_res["ramp_r2"]
                * eval_res["coherence"]
                * min(3.5, contrast_ratio / 2.0),
                4,
            )

            detected.append(
                {
                    "start_frame": int(abs_s),
                    "end_frame": int(abs_e),
                    "start_s": round(abs_s / fps, 3),
                    "end_s": round(abs_e / fps, 3),
                    "duration_s": round(dur, 3),
                    "orientation": eval_res["orientation"],
                    "direction": eval_res["direction"],
                    "action": eval_res["action"],
                    "is_multi_bar_array": bool(is_stacked_horiz or is_stacked_vert),
                    "bbox_norm": bbox_norm,
                    "monotonicity": eval_res["monotonicity"],
                    "ramp_r2": eval_res["ramp_r2"],
                    "coherence": eval_res["coherence"],
                    "strip_contrast": round(contrast_ratio, 2),
                    "quality_score": quality,
                    "song_part_mapping": {
                        "section_type": (
                            "horizontal_momentum_charge_part"
                            if use_horiz
                            else "vertical_register_climb_part"
                        ),
                        "pitch_follows_fill_curve": True,
                        "rhythm_density_follows_fill_curve": use_horiz,
                        "stereo_pan_follows_direction": eval_res["direction"],
                        "completion_accent_at_end": eval_res["action"] == "fill",
                    },
                    "fill_curve": eval_res["fill_curve"],
                }
            )

    def _bbox_iou(b1: List[float], b2: List[float]) -> float:
        ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter / max(1e-6, a1 + a2 - inter)

    detected.sort(key=lambda d: (-d["quality_score"], d["start_s"]))
    merged: List[Dict[str, Any]] = []
    for cand in detected:
        overlap = None
        for existing in merged:
            inter_s = max(cand["start_s"], existing["start_s"])
            inter_e = min(cand["end_s"], existing["end_s"])
            if (inter_e - inter_s) > 0.45 * min(cand["duration_s"], existing["duration_s"]):
                overlap = existing
                break
        if overlap is not None:
            if (
                cand["orientation"] == overlap["orientation"]
                and _bbox_iou(cand["bbox_norm"], overlap["bbox_norm"]) < 0.25
            ):
                overlap["parallel_bar_count"] = overlap.get("parallel_bar_count", 1) + 1
        else:
            cand["parallel_bar_count"] = 8 if cand.get("is_multi_bar_array") else 1
            merged.append(cand)

    merged.sort(key=lambda d: d["start_s"])

    # Stitch consecutive sub-window detections belonging to the exact same physical bar track
    stitched: List[Dict[str, Any]] = []
    for seg in merged:
        if stitched:
            prev = stitched[-1]
            same_kind = (
                seg["orientation"] == prev["orientation"]
                and seg["direction"] == prev["direction"]
                and seg["action"] == prev["action"]
            )
            b_p, b_c = prev["bbox_norm"], seg["bbox_norm"]
            if seg["orientation"] == "horizontal":
                aligned = abs(b_p[1] - b_c[1]) <= 0.025 and abs(b_p[3] - b_c[3]) <= 0.025
            else:
                aligned = abs(b_p[0] - b_c[0]) <= 0.025 and abs(b_p[2] - b_c[2]) <= 0.025
            if same_kind and aligned and (seg["start_s"] - prev["end_s"]) <= 0.40:
                u_s = min(prev["start_frame"], seg["start_frame"])
                u_e = max(prev["end_frame"], seg["end_frame"])
                ux0 = max(0, int(round(min(b_p[0], b_c[0]) * work_w)))
                uy0 = max(0, int(round(min(b_p[1], b_c[1]) * work_h)))
                ux1 = min(work_w, max(ux0 + 1, int(round(max(b_p[2], b_c[2]) * work_w))))
                uy1 = min(work_h, max(uy0 + 1, int(round(max(b_p[3], b_c[3]) * work_h))))
                axis_idx = 1 if seg["orientation"] == "horizontal" else 2
                u_prof = np.mean(sig_frames[u_s:u_e, uy0:uy1, ux0:ux1], axis=axis_idx)
                p_min = np.percentile(u_prof, 5, axis=0, keepdims=True)
                p_max = float(np.percentile(u_prof, 97))
                u_norm = np.clip((u_prof - p_min) / max(1e-5, p_max - float(np.mean(p_min))), 0.0, 1.0)
                u_frac = uniform_filter1d(np.mean(u_norm > 0.36, axis=1).astype(np.float64), size=3)
                f0, f1 = float(np.min(u_frac)), float(np.max(u_frac))
                u_curve = np.clip((u_frac - f0) / max(1e-5, f1 - f0), 0.0, 1.0)
                prev["start_frame"] = int(u_s)
                prev["end_frame"] = int(u_e)
                prev["start_s"] = round(u_s / fps, 3)
                prev["end_s"] = round(u_e / fps, 3)
                prev["duration_s"] = round((u_e - u_s) / fps, 3)
                prev["bbox_norm"] = [
                    round(ux0 / work_w, 4),
                    round(uy0 / work_h, 4),
                    round(ux1 / work_w, 4),
                    round(uy1 / work_h, 4),
                ]
                prev["monotonicity"] = round(abs(_monotonic_spearman(u_curve)), 4)
                prev["fill_curve"] = [round(float(v), 4) for v in u_curve]
                continue
        stitched.append(seg)

    return {
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": round(n_frames / fps, 3),
        "bar_fill_count": len(stitched),
        "bar_fills": stitched,
    }


def detect_video_bar_fills(
    video_path: str,
    max_frames: Optional[int] = None,
    min_duration_s: float = 0.65,
) -> Dict[str, Any]:
    """Reads a video file from disk and detects all horizontal and vertical bar fills/drains."""
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

    res = detect_frames_bar_fills(frames, fps=fps, min_duration_s=min_duration_s)
    res["video_path"] = video_path
    return res
