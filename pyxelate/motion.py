"""Computer-Vision Video Motion, Scene-Cut, Camera Zoom, and Bidirectional Music-Motion Synchronizer.

Provides two complementary capabilities:
1. Video -> Music (`analyze_frames_motion` / `analyze_video_motion`):
   Extracts frame-accurate musical synchronization events from any video clip so SoundBit MCP
   can compose a custom melodic bass-drop soundtrack where:
   - Major scene cuts trigger heavy melodic bass drops & 808/FM sub bombs.
   - Continuous camera zoom-ins & object inflations map to consonant harmonic intervals (+5, +7, +12)
     and ascending chordal arpeggio swells across the exact zoom window.
   - Continuous camera zoom-outs & object shrinkings map to consonant harmonic intervals (-5, -7, -12)
     and descending arpeggio cascades across the exact zoom-out window.
   - Frame-to-frame optical motion energy & object focal centroid drive melody rhythm & pitch contour.

2. Music -> Video Post-Refinement (`sync_frames_motion_to_audio` / `sync_video_motion_to_audio`):
   Modifies the video's camera zooms, pullbacks, focal tracking, and bass-drop micro-zoom impulses
   in high-resolution pre-pixel space to lock tightly onto the rendered WAV soundtrack's beats,
   sub-bass transients, and melodic risers/cascades before final pixel stylization.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
import imageio.v3 as iio
from .fast_ops import resize_float, rgb2hsv_fast


def _quantize_consonant_semitones(val: float, div_std: float) -> int:
    """Quantizes optical-flow zoom divergence strictly to consonant harmonic intervals (±5, ±7, ±12)."""
    ratio = abs(val) / max(1e-5, div_std)
    if ratio >= 1.85:
        mag = 12  # Pure octave
    elif ratio >= 1.35:
        mag = 7   # Perfect fifth
    else:
        mag = 5   # Perfect fourth
    return mag if val >= 0 else -mag


def analyze_frames_motion(
    frames: List[np.ndarray], fps: float = 30.0
) -> Dict[str, Any]:
    """Analyzes a sequence of RGB frames and returns frame-accurate motion & consonant musical sync events."""
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
    centroid_x = np.zeros(n_frames, dtype=np.float64)
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
        centroid_x[i] = (cx + 1.0) * 0.5  # 0.0 = left, 1.0 = right
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

    # Zero out optical-flow spikes right on hard scene cuts so cut discontinuities aren't misread as zooms
    cut_thresh = max(0.32, float(np.mean(cut_scores) + 1.85 * np.std(cut_scores)))

    # 1. Detect major Scene Cuts (for Bass Drops & Sub Bombs)
    scene_cuts: List[Dict[str, Any]] = []
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

    clean_div = radial_divergence.copy()
    for c in scene_cuts:
        cf = c["frame"]
        clean_div[max(0, cf - 1) : min(n_frames, cf + 2)] = 0.0

    # Normalize motion energy to [0, 1]
    if np.max(motion_energy) > 1e-6:
        motion_energy_norm = np.clip(
            motion_energy / max(1e-6, float(np.percentile(motion_energy, 95))) * 0.85, 0.0, 1.0
        )
    else:
        motion_energy_norm = motion_energy

    # 2. Detect continuous Zoom-In / Inflation vs Zoom-Out / Shrinking windows
    win = max(3, int(fps * 0.35))
    kernel = np.ones(win, dtype=np.float64) / win
    smooth_div = np.convolve(clean_div, kernel, mode="same")

    scale_events: List[Dict[str, Any]] = []
    div_std = max(1e-4, float(np.std(smooth_div)))
    min_scale_gap = max(10, int(fps * 0.65))
    last_scale_frame = -min_scale_gap
    cut_frames = sorted([c["frame"] for c in scene_cuts])

    for i in range(3, n_frames - 3):
        if any(abs(i - cf) <= 3 for cf in cut_frames):
            continue
        val = smooth_div[i]
        if abs(val) > 1.08 * div_std and (i - last_scale_frame) >= min_scale_gap:
            is_local_extremum = (
                val > 0 and val >= smooth_div[i - 1] and val >= smooth_div[i + 1]
            ) or (val < 0 and val <= smooth_div[i - 1] and val <= smooth_div[i + 1])
            if is_local_extremum:
                # Expand left & right to find the continuous zoom gesture window within the current shot
                prev_cut = max([0] + [cf for cf in cut_frames if cf < i])
                next_cut = min([n_frames - 1] + [cf - 1 for cf in cut_frames if cf > i])
                sign = 1.0 if val > 0 else -1.0
                active_thresh = 0.32 * div_std

                s_f = i
                while s_f > prev_cut and (smooth_div[s_f - 1] * sign) > active_thresh:
                    s_f -= 1
                e_f = i
                while e_f < next_cut and (smooth_div[e_f + 1] * sign) > active_thresh:
                    e_f += 1

                # Guarantee at least a musical quarter-note window (~0.35s)
                if (e_f - s_f) < int(fps * 0.35):
                    half = max(3, int(fps * 0.2))
                    s_f = max(prev_cut, i - half)
                    e_f = min(next_cut, i + half)

                kind = "zoom_in_or_inflate" if val > 0 else "zoom_out_or_shrink"
                semitone_slide = _quantize_consonant_semitones(float(val), div_std)
                scale_events.append(
                    {
                        "frame": int(i),
                        "start_frame": int(s_f),
                        "end_frame": int(e_f),
                        "time_s": round(i / fps, 3),
                        "start_time_s": round(s_f / fps, 3),
                        "end_time_s": round(e_f / fps, 3),
                        "duration_s": round(max(1, e_f - s_f) / fps, 3),
                        "type": kind,
                        "divergence": round(float(val), 4),
                        "suggested_slide_semitones": semitone_slide,
                        "musical_gesture": (
                            "ascending_arp_swell" if val > 0 else "descending_arp_cascade"
                        ),
                        "velocity_arc": [74, 120] if val > 0 else [118, 74],
                        "preset": "sfx_zoom_in_riser"
                        if val > 1.55 * div_std
                        else (
                            "sfx_object_inflate"
                            if val > 0
                            else (
                                "sfx_zoom_out_fall"
                                if val < -1.55 * div_std
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
            "centroid_x": [round(float(x), 4) for x in centroid_x],
            "centroid_y": [round(float(x), 4) for x in centroid_y],
            "radial_divergence": [round(float(x), 4) for x in smooth_div],
            "brightness": [round(float(x), 4) for x in brightness],
        },
    }


def analyze_video_motion(video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Reads a video file from disk and runs full motion, scene-cut, and zoom/scale analysis."""
    fps = 30.0
    try:
        meta = iio.immeta(video_path)
        fps = float(meta.get("fps", 30.0))
    except Exception:
        fps = 30.0

    frames = []
    for frame in iio.imiter(video_path):
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    result = analyze_frames_motion(frames, fps=fps)
    result["video_path"] = video_path
    return result


def _warp_zoom_frame(
    frame: np.ndarray, zoom: float, focus_offset_x: float = 0.0, focus_offset_y: float = 0.0
) -> np.ndarray:
    """Fast separable sub-pixel bilinear zoom crop toward `(focus_offset_x, focus_offset_y)` in [-1, 1]."""
    if zoom <= 1.0008:
        return frame
    arr = np.asarray(frame)
    is_u8 = arr.dtype == np.uint8
    f = arr[:, :, :3].astype(np.float32)
    h, w, _ = f.shape

    ch = h / float(zoom)
    cw = w / float(zoom)
    max_dx = (w - cw) * 0.5
    max_dy = (h - ch) * 0.5

    cx = w * 0.5 + float(np.clip(focus_offset_x, -1.0, 1.0)) * max_dx * 0.85
    cy = h * 0.5 + float(np.clip(focus_offset_y, -1.0, 1.0)) * max_dy * 0.85

    x0 = float(np.clip(cx - cw * 0.5, 0.0, max(0.0, w - cw)))
    y0 = float(np.clip(cy - ch * 0.5, 0.0, max(0.0, h - ch)))

    ys = np.clip(np.linspace(y0, y0 + ch - 1.0, h, dtype=np.float32), 0.0, h - 1.0001)
    xs = np.clip(np.linspace(x0, x0 + cw - 1.0, w, dtype=np.float32), 0.0, w - 1.0001)

    iy0 = ys.astype(np.int32)
    iy1 = np.minimum(iy0 + 1, h - 1)
    wy = (ys - iy0)[:, None, None]
    row_interp = (1.0 - wy) * f[iy0] + wy * f[iy1]

    ix0 = xs.astype(np.int32)
    ix1 = np.minimum(ix0 + 1, w - 1)
    wx = (xs - ix0)[None, :, None]
    out = (1.0 - wx) * row_interp[:, ix0] + wx * row_interp[:, ix1]

    if is_u8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def _extract_audio_sync_envelopes(
    audio_wav_path: str, n_frames: int, fps: float
) -> Dict[str, np.ndarray]:
    """Extracts frame-accurate sub-bass kick/drop punch and melodic swell envelopes from a WAV track."""
    sr, data = wavfile.read(audio_wav_path)
    if data.dtype == np.int16:
        sig = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        sig = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        sig = (data.astype(np.float64) - 128.0) / 128.0
    else:
        sig = data.astype(np.float64)

    if sig.ndim > 1:
        mono = np.mean(sig, axis=1)
    else:
        mono = sig

    nyq = 0.5 * sr
    sos_low = butter(4, min(0.45, 135.0 / nyq), btype="low", output="sos")
    sos_mid = butter(
        4,
        [min(0.4, 380.0 / nyq), min(0.92, 6500.0 / nyq)],
        btype="band",
        output="sos",
    )
    low_band = sosfiltfilt(sos_low, mono)
    mid_band = sosfiltfilt(sos_mid, mono)

    samples_per_frame = sr / float(fps)
    bass_rms = np.zeros(n_frames, dtype=np.float64)
    mid_rms = np.zeros(n_frames, dtype=np.float64)

    for f in range(n_frames):
        s0 = int(round(f * samples_per_frame))
        s1 = min(len(mono), int(round((f + 1) * samples_per_frame)))
        if s1 > s0:
            bass_rms[f] = float(np.sqrt(np.mean(low_band[s0:s1] ** 2)))
            mid_rms[f] = float(np.sqrt(np.mean(mid_band[s0:s1] ** 2)))

    # Normalize smooth melodic energy
    mid_smooth = gaussian_filter1d(mid_rms, sigma=2.5)
    mid_p95 = max(1e-5, float(np.percentile(mid_smooth, 96)))
    melodic_env = np.clip(mid_smooth / mid_p95, 0.0, 1.0)

    # Extract sharp bass-drop & kick transient impulses with smooth musical eighth-note decay
    onset = np.zeros(n_frames, dtype=np.float64)
    for f in range(1, n_frames):
        onset[f] = max(0.0, bass_rms[f] - 0.72 * bass_rms[f - 1])
    onset[0] = bass_rms[0]
    onset_p97 = max(1e-5, float(np.percentile(onset, 97)))
    onset_norm = np.clip(onset / onset_p97, 0.0, 1.25)

    bass_punch = np.zeros(n_frames, dtype=np.float64)
    cur = 0.0
    decay = float(np.exp(-1.0 / max(1.5, fps * 0.14)))  # ~140ms smooth musical release
    for f in range(n_frames):
        cur = max(onset_norm[f], cur * decay)
        bass_punch[f] = cur
    # 1-frame micro-smoothing so transient hits feel punchy yet silky
    bass_punch = gaussian_filter1d(bass_punch, sigma=0.65)

    return {
        "bass_rms": bass_rms,
        "melodic_env": melodic_env,
        "bass_punch": np.clip(bass_punch, 0.0, 1.2),
    }


def _ease_in_out_cubic(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 4.0 * t * t * t, 1.0 - np.power(-2.0 * t + 2.0, 3.0) * 0.5)


def sync_frames_motion_to_audio(
    frames: List[np.ndarray],
    audio_wav_path: str,
    fps: float = 30.0,
    bpm: float = 120.0,
    motion_analysis: Optional[Dict[str, Any]] = None,
    phrase_zoom_strength: float = 0.16,
    bass_punch_strength: float = 0.062,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Refines video motion & camera zooms in high-res space to lock onto the music WAV track.

    For each shot (bounded by `scene_cuts`):
    - Aligns zoom-in swells & zoom-out pullbacks to the musical beat/bar phrase and melodic envelope
      using smooth cubic easing (replacing erratic or off-beat camera zooms with beat-locked motion).
    - Adds tight sub-bass / kick kinetic micro-zoom pulses synced to actual 20-135 Hz WAV transients.
    - Tracks each shot's smoothed salience focal point so zooms pull cleanly into the main subject.
    """
    n_frames = len(frames)
    if n_frames == 0:
        return [], {"n_frames": 0}

    analysis = motion_analysis or analyze_frames_motion(frames, fps=fps)
    audio_env = _extract_audio_sync_envelopes(audio_wav_path, n_frames=n_frames, fps=fps)
    melodic_env = audio_env["melodic_env"]
    bass_punch = audio_env["bass_punch"]

    cut_frames = sorted(
        list({0, n_frames} | {int(c["frame"]) for c in analysis.get("scene_cuts", []) if 0 < int(c["frame"]) < n_frames})
    )
    raw_div = np.asarray(analysis["per_frame"]["radial_divergence"], dtype=np.float64)
    raw_cx = np.asarray(
        analysis["per_frame"].get("centroid_x", [0.5] * n_frames), dtype=np.float64
    )
    raw_cy = np.asarray(analysis["per_frame"]["centroid_y"], dtype=np.float64)

    frames_per_beat = (fps * 60.0) / max(30.0, float(bpm))
    phrase_zoom = np.zeros(n_frames, dtype=np.float64)
    focus_ox = np.zeros(n_frames, dtype=np.float64)
    focus_oy = np.zeros(n_frames, dtype=np.float64)
    shot_summaries: List[Dict[str, Any]] = []

    for idx in range(len(cut_frames) - 1):
        s0, s1 = cut_frames[idx], cut_frames[idx + 1]
        length = s1 - s0
        if length <= 0:
            continue

        # Smooth focal subject offset within this shot (clamped to [-0.45, +0.45] so framing stays balanced)
        shot_ox = gaussian_filter1d((raw_cx[s0:s1] - 0.5) * 2.0, sigma=max(2.0, fps * 0.30))
        shot_oy = gaussian_filter1d((0.5 - raw_cy[s0:s1]) * 2.0, sigma=max(2.0, fps * 0.30))
        focus_ox[s0:s1] = np.clip(shot_ox, -0.45, 0.45)
        focus_oy[s0:s1] = np.clip(shot_oy, -0.45, 0.45)

        shot_div = float(np.mean(raw_div[s0:s1]))
        mel_slope = (
            float(np.mean(melodic_env[s0 + length // 2 : s1]) - np.mean(melodic_env[s0 : s0 + max(1, length // 2)]))
            if length >= 6
            else 0.0
        )

        t_norm = np.linspace(0.0, 1.0, length, dtype=np.float64)
        eased = _ease_in_out_cubic(t_norm)
        # Beat-locked breathing modulation (peaks gently on musical beats/half-bars)
        beat_phase = (np.arange(s0, s1, dtype=np.float64) / frames_per_beat) * (2.0 * np.pi / 2.0)
        beat_breath = 0.025 * (0.5 - 0.5 * np.cos(beat_phase)) * melodic_env[s0:s1]

        # Decide musical zoom gesture for this shot from combined visual divergence + musical phrase slope
        combined_score = shot_div * 1.4 + mel_slope * 0.45
        if combined_score >= 0.015:
            mode = "musical_zoom_in_swell"
            arc = phrase_zoom_strength * (0.20 + 0.80 * eased) * (0.65 + 0.35 * melodic_env[s0:s1])
        elif combined_score <= -0.015:
            mode = "musical_zoom_out_release"
            arc = phrase_zoom_strength * (0.92 * (1.0 - eased)) * (0.70 + 0.30 * melodic_env[s0:s1])
        else:
            # Alternate subtle musical push-in / pull-out across shots so motion always flows with the bars
            if idx % 2 == 0:
                mode = "bar_locked_push_in"
                arc = phrase_zoom_strength * 0.65 * eased
            else:
                mode = "bar_locked_pull_out"
                arc = phrase_zoom_strength * 0.60 * (1.0 - eased)

        phrase_zoom[s0:s1] = arc + beat_breath
        shot_summaries.append(
            {
                "shot_index": idx,
                "start_frame": int(s0),
                "end_frame": int(s1),
                "mode": mode,
                "start_zoom": round(1.0 + float(phrase_zoom[s0]), 4),
                "end_zoom": round(1.0 + float(phrase_zoom[s1 - 1]), 4),
            }
        )

    total_zoom = np.clip(1.0 + phrase_zoom + bass_punch_strength * bass_punch, 1.0, 1.28)

    synced_frames: List[np.ndarray] = []
    for f in range(n_frames):
        synced_frames.append(
            _warp_zoom_frame(
                frames[f],
                zoom=float(total_zoom[f]),
                focus_offset_x=float(focus_ox[f]),
                focus_offset_y=float(focus_oy[f]),
            )
        )

    telemetry = {
        "n_frames": n_frames,
        "fps": fps,
        "bpm": bpm,
        "max_zoom": round(float(np.max(total_zoom)), 4),
        "mean_zoom": round(float(np.mean(total_zoom)), 4),
        "shots_synced": shot_summaries,
        "per_frame_zoom": [round(float(z), 4) for z in total_zoom],
        "per_frame_bass_punch": [round(float(b), 4) for b in bass_punch],
    }
    return synced_frames, telemetry


def sync_video_motion_to_audio(
    input_video: str,
    audio_wav: str,
    output_video: str,
    bpm: float = 120.0,
    style: Optional[str] = "pc98_cyber",
    factor: int = 3,
    phrase_zoom_strength: float = 0.16,
    bass_punch_strength: float = 0.062,
) -> Dict[str, Any]:
    """End-to-end helper: synchronizes video motion/zooms to `audio_wav`, applies pixel style, & muxes MP4."""
    import shutil
    import subprocess
    from pathlib import Path
    from .styles import apply_style

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    fps = 30.0
    try:
        fps = float(iio.immeta(input_video).get("fps", 30.0))
    except Exception:
        fps = 30.0

    raw_frames = [f for f in iio.imiter(input_video)]
    synced_frames, telemetry = sync_frames_motion_to_audio(
        raw_frames,
        audio_wav_path=audio_wav,
        fps=fps,
        bpm=bpm,
        phrase_zoom_strength=phrase_zoom_strength,
        bass_punch_strength=bass_punch_strength,
    )

    if style:
        out_frames = [
            apply_style(frame, style=style, factor=factor, upscale=factor, frame_index=idx)
            for idx, frame in enumerate(synced_frames)
        ]
    else:
        out_frames = synced_frames

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_vid = str(out_path.with_suffix(".silent_sync.mp4"))
    iio.imwrite(tmp_vid, out_frames, fps=fps, codec="libx264")

    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            tmp_vid,
            "-i",
            audio_wav,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(tmp_vid).unlink(missing_ok=True)
    return {
        "output_video": str(out_path),
        "style": style or "none",
        "frames": len(out_frames),
        "fps": fps,
        "bpm": bpm,
        "max_zoom": telemetry["max_zoom"],
        "shots_synced": len(telemetry["shots_synced"]),
    }
