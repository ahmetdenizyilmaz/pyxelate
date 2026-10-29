"""Generalized Video Movement-to-Sound & Musical Regime Trigger Extractor.

Implements domain-independent computer-vision detectors for 8 universal visual-to-audio
trigger families and 5 automatic musical arrangement regimes discovered via DSP reverse-engineering,
applicable to any science explainer, 3D/2D mechanism, UI, documentary, or pixel-art video.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import imageio.v3 as iio

from .fast_ops import resize_float, rgb2hsv_fast
from .motion import analyze_frames_motion


CATALOG_PATH = Path(__file__).with_name("generalized_sound_triggers.json")


def load_trigger_catalog() -> Dict[str, Any]:
    """Loads the declarative JSON specification of generalized movement-to-sound triggers."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_generalized_triggers(
    frames: List[np.ndarray], fps: float = 30.0
) -> Dict[str, Any]:
    """Extracts generalized visual sound triggers and musical regimes from any RGB frame sequence."""
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames provided")

    base = analyze_frames_motion(frames, fps=fps)
    dur_s = n_frames / fps

    work_h, work_w = 128, 96
    dark_void_ratio = np.zeros(n_frames, dtype=np.float64)
    midtone_surface_ratio = np.zeros(n_frames, dtype=np.float64)
    hud_bright_mono = np.zeros(n_frames, dtype=np.float64)
    emissive_chroma = np.zeros(n_frames, dtype=np.float64)
    indicator_badge = np.zeros(n_frames, dtype=np.float64)
    warm_subject_area = np.zeros(n_frames, dtype=np.float64)
    isolated_particle_score = np.zeros(n_frames, dtype=np.float64)

    for i, raw in enumerate(frames):
        rgb = resize_float(raw, work_h, work_w, anti_aliasing=True)
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hsv = rgb2hsv_fast(rgb)
        sat = hsv[:, :, 1]

        dark_void_ratio[i] = float(np.mean(luma < 0.11))
        # Connected structural housings/rooms typically occupy midtone low-to-medium saturation surfaces
        midtone_surface_ratio[i] = float(np.mean((luma >= 0.13) & (luma <= 0.62) & (sat < 0.48)))

        # HUD/caption text: bright monochrome pixels in upper 28% or lower 18% overlay bands
        top_band = (luma[: int(work_h * 0.28), :] > 0.80) & (sat[: int(work_h * 0.28), :] < 0.16)
        hud_bright_mono[i] = float(np.mean(top_band))

        # High-chroma luminous beams, sparks, plasma, or active energy highlights
        emissive = (luma > 0.48) & (sat > 0.52)
        emissive_chroma[i] = float(np.mean(emissive))

        # Compact status indicators (green/amber/cyan glowing badges covering < 5% of frame)
        badge_mask = (luma > 0.52) & (sat > 0.60)
        badge_frac = float(np.mean(badge_mask))
        indicator_badge[i] = badge_frac if badge_frac < 0.06 else 0.0

        # Active warm/highlighted cluster mass (tracks growing or shrinking target structures/cells)
        warm_subject_area[i] = float(np.mean((rgb[:, :, 0] > 0.52) & (sat > 0.38)))

        # Particle sparsity: high local contrast dots surrounded by dark void
        gy, gx = np.gradient(luma)
        high_freq = float(np.mean(np.sqrt(gx * gx + gy * gy) > 0.12))
        isolated_particle_score[i] = (
            high_freq * dark_void_ratio[i] / max(0.05, midtone_surface_ratio[i] + 0.05)
        )

    mot = np.asarray(base["per_frame"]["motion_energy"], dtype=np.float64)
    cuts = sorted([int(c["frame"]) for c in base["scene_cuts"]])
    shot_bounds = sorted(list({0, n_frames} | {c for c in cuts if 0 < c < n_frames}))

    events: List[Dict[str, Any]] = []

    # 1. Camera Zoom-In & Zoom-Out Continuous Sweep Triggers
    for sc in base["scale_events"]:
        is_in = sc["type"] == "zoom_in_or_inflate"
        events.append(
            {
                "trigger": "camera_zoom_in" if is_in else "camera_zoom_out",
                "time_s": sc["start_time_s"],
                "end_s": sc["end_time_s"],
                "duration_s": sc["duration_s"],
                "slide_semitones": 12 if is_in else -12,
                "preset": "trig_zoom_sweep_pure",
                "layer_preset": "trig_whoosh_air_burst",
                "intensity": round(min(1.0, abs(sc["divergence"]) * 8.0 + 0.45), 3),
            }
        )

    # 2. HUD Text / Telemetry Typing Ticks
    hud_delta = np.maximum(0.0, np.diff(hud_bright_mono, prepend=hud_bright_mono[0]))
    # Zero out hard scene cut frames so full-card cuts aren't misread as typing
    for cf in cuts:
        hud_delta[max(0, cf - 1) : min(n_frames, cf + 2)] = 0.0
    hud_thresh = max(0.0008, float(np.percentile(hud_delta, 91)))
    hud_peaks, _ = find_peaks(hud_delta, height=hud_thresh, distance=max(2, int(fps * 0.09)))
    for p in hud_peaks:
        events.append(
            {
                "trigger": "hud_text_typing",
                "time_s": round(float(p / fps), 3),
                "duration_s": 0.06,
                "preset": "trig_ui_text_tick",
                "intensity": round(float(np.clip(hud_delta[p] / (hud_thresh * 2.5), 0.45, 1.0)), 3),
            }
        )

    # 3. UI Status Indicator / Button Activation Chimes
    badge_delta = np.maximum(0.0, np.diff(indicator_badge, prepend=indicator_badge[0]))
    for cf in cuts:
        badge_delta[max(0, cf - 1) : min(n_frames, cf + 2)] = 0.0
    b_thresh = max(0.0006, float(np.percentile(badge_delta, 96.5)))
    b_peaks, _ = find_peaks(badge_delta, height=b_thresh, distance=max(8, int(fps * 0.45)))
    for p in b_peaks:
        events.append(
            {
                "trigger": "state_confirm_chime",
                "time_s": round(float(p / fps), 3),
                "duration_s": 0.25,
                "interval_semitones": 7,
                "preset": "trig_confirm_chime",
                "intensity": 0.85,
            }
        )

    # 4. Energy Beam / Emission Bursts
    em_smooth = gaussian_filter1d(emissive_chroma, sigma=1.5)
    em_delta = np.maximum(0.0, np.diff(em_smooth, prepend=em_smooth[0]))
    for cf in cuts:
        em_delta[max(0, cf - 1) : min(n_frames, cf + 2)] = 0.0
    em_thresh = max(0.0007, float(np.percentile(em_delta, 96)))
    em_peaks, _ = find_peaks(em_delta, height=em_thresh, distance=max(12, int(fps * 0.7)))
    for p in em_peaks:
        events.append(
            {
                "trigger": "energy_emission_burst",
                "time_s": round(float(p / fps), 3),
                "duration_s": 0.35,
                "preset": "trig_whoosh_air_burst",
                "sub_impact": True,
                "intensity": round(float(np.clip(em_delta[p] / (em_thresh * 2.2), 0.5, 1.0)), 3),
            }
        )

    # 5. Shot-Level Regime Classification + Sustained Acceleration Risers + Mechanical Step Trains!
    regimes: List[Dict[str, Any]] = []
    first_establish_end = min(max(4.0, dur_s * 0.075), 9.0)

    for idx in range(len(shot_bounds) - 1):
        s0, s1 = shot_bounds[idx], shot_bounds[idx + 1]
        if s1 - s0 < 4:
            continue
        t0, t1 = s0 / fps, s1 / fps
        shot_len_s = t1 - t0

        m_void = float(np.mean(dark_void_ratio[s0:s1]))
        m_surf = float(np.mean(midtone_surface_ratio[s0:s1]))
        m_part = float(np.mean(isolated_particle_score[s0:s1]))
        m_mot = float(np.mean(mot[s0:s1]))

        # Classify musical arrangement regime from pure spatial/temporal visual topology
        if t1 <= first_establish_end:
            regime = "intro_ambient"
        elif t0 >= dur_s - max(3.0, dur_s * 0.045) and m_mot < 0.26:
            regime = "resolution_outro"
        elif m_void >= 0.70 and m_surf < 0.18 and m_part > 0.22:
            regime = "micro_ambient_particles"
        else:
            # Check if this macro shot has sustained laminar flow / monotonic energy acceleration
            first_half_em = float(np.mean(emissive_chroma[s0 : s0 + (s1 - s0) // 2]))
            second_half_em = float(np.mean(emissive_chroma[s0 + (s1 - s0) // 2 : s1]))
            sustained_flow = float(np.percentile(mot[s0:s1], 25)) > 0.14 and m_mot > 0.20
            if shot_len_s >= 3.0 and (sustained_flow or second_half_em > first_half_em * 1.25) and m_mot > 0.18:
                regime = "macro_double_time_accel"
                events.append(
                    {
                        "trigger": "sustained_acceleration_riser",
                        "time_s": round(t0, 3),
                        "end_s": round(t1, 3),
                        "duration_s": round(shot_len_s, 3),
                        "preset": "trig_zoom_sweep_pure",
                        "pitch_climb_semitones": 12,
                        "intensity": 0.85,
                    }
                )
            else:
                regime = "macro_driving_pulse"

        regimes.append(
            {
                "shot_index": idx,
                "start_frame": int(s0),
                "end_frame": int(s1),
                "start_s": round(t0, 3),
                "end_s": round(t1, 3),
                "regime": regime,
                "dark_void_ratio": round(m_void, 3),
                "midtone_surface_ratio": round(m_surf, 3),
                "mean_motion": round(m_mot, 3),
            }
        )

        # Inside `micro_ambient_particles`, emit discrete droplet/ionization bell pings on collision peaks!
        if regime == "micro_ambient_particles":
            sub_mot = gaussian_filter1d(mot[s0:s1], sigma=1.2)
            pks, _ = find_peaks(
                sub_mot,
                prominence=max(0.03, float(np.std(sub_mot) * 0.45)),
                distance=max(5, int(fps * 0.28)),
            )
            for pk in pks:
                events.append(
                    {
                        "trigger": "micro_particle_collision",
                        "time_s": round(float((s0 + pk) / fps), 3),
                        "duration_s": 0.22,
                        "preset": "trig_micro_droplet_ping",
                        "intensity": round(float(np.clip(sub_mot[pk] * 1.3, 0.45, 1.0)), 3),
                    }
                )
        else:
            # Inside macro/grid shots, detect repeating mechanical sub-part step trains (e.g. blades, gears, cell grid pops!)
            sub_mot = gaussian_filter1d(mot[s0:s1], sigma=1.0)
            pks, _ = find_peaks(
                sub_mot,
                prominence=max(0.025, float(np.std(sub_mot) * 0.40)),
                distance=max(3, int(fps * 0.14)),
            )
            if len(pks) >= 4:
                area_start = float(np.mean(warm_subject_area[s0 : s0 + max(2, (s1 - s0) // 4)]))
                area_end = float(np.mean(warm_subject_area[s1 - max(2, (s1 - s0) // 4) : s1]))
                if area_end < area_start * 0.65 and area_start > 0.01:
                    direction = "shrinking"
                elif area_end > area_start * 1.35 and area_end > 0.01:
                    direction = "expanding"
                else:
                    direction = "oscillating"
                events.append(
                    {
                        "trigger": "mechanical_step_train",
                        "time_s": round(float((s0 + pks[0]) / fps), 3),
                        "end_s": round(float((s0 + pks[-1]) / fps), 3),
                        "step_times_s": [round(float((s0 + pk) / fps), 3) for pk in pks],
                        "direction": direction,
                        "preset": "trig_stepped_mech_pop",
                        "intensity": 0.85,
                    }
                )

    events.sort(key=lambda e: e["time_s"])
    counts: Dict[str, int] = {}
    for e in events:
        counts[e["trigger"]] = counts.get(e["trigger"], 0) + 1

    return {
        "schema": "pyxelate-extracted-video-triggers-v1",
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": round(dur_s, 3),
        "trigger_counts": counts,
        "regimes": regimes,
        "scene_cuts": base["scene_cuts"],
        "events": events,
    }


def extract_video_sound_triggers(
    video_path: str, max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """Reads any video file and extracts its generalized movement-to-sound triggers & musical regimes."""
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

    res = extract_generalized_triggers(frames, fps=fps)
    res["video_path"] = video_path
    return res
