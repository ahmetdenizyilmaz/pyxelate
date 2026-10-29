"""Computer-Vision Video Motion, Scene-Cut, Camera Zoom, and Object Inflation/Shrinking Analyzer.

Extracts frame-accurate musical synchronization events from any video clip so SoundBit MCP
can compose a custom melodic bass-drop soundtrack where:
- Major scene cuts trigger heavy melodic bass drops & 808 sub bombs.
- Camera zoom-ins & object inflations trigger upward Hz/semitone pitch glides & rising arpeggios.
- Camera zoom-outs & object shrinkings trigger downward pitch glides & contracting plucks.
- Frame-to-frame optical motion energy & vertical object position drive melody rhythm & pitch contour.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import imageio.v3 as iio
from .fast_ops import resize_float, rgb2hsv_fast


def analyze_frames_motion(
    frames: List[np.ndarray], fps: float = 30.0
) -> Dict[str, Any]:
    """Analyzes a sequence of RGB frames and returns frame-accurate motion & musical sync events."""
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames provided for motion analysis")

    work_h, work_w = 90, 160
    small_frames = [resize_float(f, work_h, work_w, anti_aliasing=True) for f in frames]

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, work_h), np.linspace(-1.0, 1.0, work_w), indexing="ij"
    )

    cut_scores = np.zeros(n_frames, dtype=np.float64)
    motion_energy = np.zeros(n_frames, dtype=np.float64)
    radial_divergence = np.zeros(n_frames, dtype=np.float64)
    object_radius = np.zeros(n_frames, dtype=np.float64)
    centroid_y = np.zeros(n_frames, dtype=np.float64)
    brightness = np.zeros(n_frames, dtype=np.float64)

    prev_luma = None
    prev_hist = None

    for i, rgb in enumerate(small_frames):
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hsv = rgb2hsv_fast(rgb)
        sat = hsv[:, :, 1]
        brightness[i] = float(np.mean(luma))

        # Salience weight (edges + saturation + brightness contrast)
        gy, gx = np.gradient(luma)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        salience = np.clip(grad_mag * 3.5 + sat * 0.45, 1e-4, None)
        sal_sum = float(np.sum(salience))

        cy = float(np.sum(yy * salience) / sal_sum)
        cx = float(np.sum(xx * salience) / sal_sum)
        centroid_y[i] = (1.0 - cy) * 0.5  # 1.0 = top of frame (high pitch), 0.0 = bottom (low pitch)

        # Effective object radius (second moment around centroid)
        ry = yy - cy
        rx = xx - cx
        r_dist = np.sqrt(rx * rx + ry * ry) + 1e-4
        object_radius[i] = float(np.sum(r_dist * salience) / sal_sum)

        # 3D RGB histogram for scene cut detection
        hist, _ = np.histogramdd(
            rgb.reshape(-1, 3), bins=(6, 6, 6), range=((0, 1), (0, 1), (0, 1)), density=True
        )
        hist = hist.ravel()

        if prev_luma is not None and prev_hist is not None:
            dt = luma - prev_luma
            abs_diff = float(np.mean(np.abs(dt)))
            hist_diff = float(0.5 * np.sum(np.abs(hist - prev_hist)))
            cut_scores[i] = 0.55 * hist_diff + 0.45 * (abs_diff * 3.2)
            motion_energy[i] = float(np.mean(np.abs(dt)) * 4.5 + np.mean(grad_mag) * 1.2)

            # Radial optical flow divergence relative to object centroid:
            # v_r = -dt * (gx * rx_unit + gy * ry_unit) / (gx^2 + gy^2 + eps)
            rx_u = rx / r_dist
            ry_u = ry / r_dist
            g_radial = gx * rx_u + gy * ry_u
            denom = gx * gx + gy * gy + 0.008
            v_radial = (-dt * g_radial) / denom
            # Weight by gradient magnitude so flat backgrounds don't add noise
            edge_weight = np.clip(grad_mag * 6.0, 0.0, 1.0)
            flow_div = float(np.sum(v_radial * edge_weight) / (np.sum(edge_weight) + 1e-4))
            radius_delta = float(object_radius[i] - object_radius[i - 1])
            radial_divergence[i] = flow_div * 4.5 + radius_delta * 6.0

        prev_luma = luma
        prev_hist = hist

    # Normalize motion energy to [0, 1]
    if np.max(motion_energy) > 1e-6:
        motion_energy_norm = np.clip(
            motion_energy / np.percentile(motion_energy, 95) * 0.85, 0.0, 1.0
        )
    else:
        motion_energy_norm = motion_energy

    # 1. Detect major Scene Cuts (for Bass Drops & Sub Bombs)
    scene_cuts: List[Dict[str, Any]] = []
    cut_thresh = max(0.32, float(np.mean(cut_scores) + 1.85 * np.std(cut_scores)))
    min_gap_frames = max(12, int(fps * 0.85))
    last_cut = -min_gap_frames
    for i in range(2, n_frames - 2):
        if cut_scores[i] >= cut_thresh and (i - last_cut) >= min_gap_frames:
            if cut_scores[i] >= cut_scores[i - 1] and cut_scores[i] >= cut_scores[i + 1]:
                intensity = float(np.clip(cut_scores[i] / (cut_thresh * 1.6), 0.55, 1.0))
                scene_cuts.append(
                    {
                        "frame": int(i),
                        "time_s": round(i / fps, 3),
                        "score": round(float(cut_scores[i]), 3),
                        "intensity": round(intensity, 3),
                        "musical_action": "bass_drop_impact",
                    }
                )
                last_cut = i

    # 2. Detect Zoom-In / Object Inflation vs Zoom-Out / Object Shrinking windows
    # Smooth radial_divergence over a ~0.35s window to capture coherent zoom/scale gestures
    win = max(3, int(fps * 0.35))
    kernel = np.ones(win, dtype=np.float64) / win
    smooth_div = np.convolve(radial_divergence, kernel, mode="same")

    scale_events: List[Dict[str, Any]] = []
    div_std = max(1e-4, float(np.std(smooth_div)))
    min_scale_gap = max(10, int(fps * 0.65))
    last_scale_frame = -min_scale_gap

    for i in range(3, n_frames - 3):
        # Ignore frames right on a hard scene cut (cut discontinuity isn't a smooth zoom)
        if any(abs(i - c["frame"]) <= 2 for c in scene_cuts):
            continue
        val = smooth_div[i]
        if abs(val) > 1.15 * div_std and (i - last_scale_frame) >= min_scale_gap:
            is_local_extremum = (
                val > 0 and val >= smooth_div[i - 1] and val >= smooth_div[i + 1]
            ) or (val < 0 and val <= smooth_div[i - 1] and val <= smooth_div[i + 1])
            if is_local_extremum:
                kind = "zoom_in_or_inflate" if val > 0 else "zoom_out_or_shrink"
                semitone_slide = int(np.clip(round((val / (div_std * 2.2)) * 12), -19, 19))
                if semitone_slide == 0:
                    semitone_slide = 7 if val > 0 else -7
                scale_events.append(
                    {
                        "frame": int(i),
                        "time_s": round(i / fps, 3),
                        "type": kind,
                        "divergence": round(float(val), 4),
                        "suggested_slide_semitones": semitone_slide,
                        "preset": "sfx_zoom_in_riser"
                        if val > 1.65 * div_std
                        else (
                            "sfx_object_inflate"
                            if val > 0
                            else (
                                "sfx_zoom_out_fall"
                                if val < -1.65 * div_std
                                else "sfx_object_shrink"
                            )
                        ),
                    }
                )
                last_scale_frame = i

    duration_s = n_frames / fps
    return {
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": round(duration_s, 3),
        "scene_cuts": scene_cuts,
        "scale_events": scale_events,
        "per_frame": {
            "motion_energy": [round(float(x), 4) for x in motion_energy_norm],
            "centroid_y": [round(float(x), 4) for x in centroid_y],
            "radial_divergence": [round(float(x), 4) for x in smooth_div],
            "brightness": [round(float(x), 4) for x in brightness],
        },
    }


def analyze_video_motion(video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Reads a video file from disk and runs full motion, scene-cut, and zoom/scale analysis."""
    props = iio.improps(video_path, plugin="pyav") if hasattr(iio, "improps") else None
    fps = 30.0
    try:
        meta = iio.immeta(video_path)
        fps = float(meta.get("fps", 30.0))
    except Exception:
        fps = 30.0

    frames = []
    for idx, frame in enumerate(iio.imiter(video_path)):
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    result = analyze_frames_motion(frames, fps=fps)
    result["video_path"] = video_path
    return result
