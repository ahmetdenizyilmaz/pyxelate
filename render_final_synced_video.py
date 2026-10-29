"""Renders the Final Motion-Synced Stylized NASA Solar Documentary Video with SoundBit Audio."""

import json
import subprocess
import time
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw

from pyxelate.styles import apply_style

OUT_DIR = Path(r"C:\Users\ady\.gemini\antigravity-cli\brain\724b9fbc-b2e4-4d9e-ac06-547dbf4ef32e")
SCRATCH_DIR = OUT_DIR / "scratch"
FFMPEG_EXE = r"C:\Users\ady\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"

DOC_VIDEO = SCRATCH_DIR / "nasa_solar_documentary_cut.mp4"
ANALYSIS_JSON = SCRATCH_DIR / "doc_motion_analysis.json"
AUDIO_WAV = OUT_DIR / "solar-probe-motion-sync.wav"
FINAL_MP4 = OUT_DIR / "solar_probe_pixel_sync_final.mp4"
PREVIEW_GRID_PNG = OUT_DIR / "solar_probe_synced_keyframes.png"

# Style rotation across the documentary chapters so the user sees pixel art + Kurzgesagt + Oscilloscope in action
SCENE_STYLES = [
    (0, "kurzgesagt_glow", "KURZGESAGT COSMIC GLOW"),
    (118, "bitwonder_bayer", "BITWONDER 16-BIT BAYER PIXEL"),
    (208, "chunky_cartoon", "CHUNKY CEL-SHADED TOON"),
    (250, "pc98_cyber", "NEC PC-98 CYBER PIXEL"),
    (299, "kurzgesagt_glow", "KURZGESAGT BIOLUMINESCENT"),
    (420, "aloboi_dreamcore", "ALOBOI ANALOG VHS PIXEL"),
    (540, "thermal_synthwave", "THERMAL SYNTHWAVE PIXEL"),
    (630, "bitwonder_bayer", "BITWONDER 16-BIT FINALE"),
]

BAR_NOTES = [
    ["D5", "F5", "E5", "D5", "A4", "D5", "C5", "A4"],
    ["Bb4", "D5", "F5", "G5", "F5", "D5", "Bb4", "F4"],
    ["G4", "Bb4", "D5", "Bb4", "F5", "D5", "Bb4", "G4"],
    ["A4", "C#5", "E5", "G5", "F5", "E5", "C#5", "A4"],
]

NOTE_HZ = {
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "Bb4": 466.16,
    "C5": 523.25,
    "C#5": 554.37,
    "D5": 587.33,
    "E5": 659.25,
    "F5": 698.46,
    "G5": 783.99,
}


def get_active_style(frame_idx: int):
    active = SCENE_STYLES[0]
    for start_f, s_id, s_label in SCENE_STYLES:
        if frame_idx >= start_f:
            active = (start_f, s_id, s_label)
    return active[1], active[2]


def main():
    t0 = time.perf_counter()
    with open(ANALYSIS_JSON, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    cut_frames = {c["frame"]: c for c in analysis["scene_cuts"]}
    cut_frames[0] = {"frame": 0, "time_s": 0.0, "intensity": 1.0}
    scale_frames = {s["frame"]: s for s in analysis["scale_events"]}

    raw_frames = [f for f in iio.imiter(DOC_VIDEO)]
    n_frames = len(raw_frames)
    print(f"Stylizing {n_frames} frames with motion-synced HUD...")

    rendered_frames = []
    keyframe_samples = []

    for idx, frame in enumerate(raw_frames):
        style_id, style_label = get_active_style(idx)
        styled = apply_style(frame, style=style_id, factor=3, upscale=3, frame_index=idx)

        # Check if we are within 10 frames (~0.33s) of a major Scene-Cut Bass Drop
        recent_cut = None
        for cf, cdata in cut_frames.items():
            if 0 <= (idx - cf) <= 10:
                recent_cut = (idx - cf, cdata)
                break

        # Check if we are within 14 frames (~0.46s) of a Zoom-In/Inflation or Zoom-Out/Shrinking event
        recent_scale = None
        for sf, sdata in scale_frames.items():
            if -2 <= (idx - sf) <= 14:
                recent_scale = (idx - sf, sdata)
                break

        # Current musical bar & step at 120 BPM (60 frames/bar, 3.75 frames/step)
        bar = min(11, idx // 60)
        step_in_bar = min(15, int((idx % 60) / 3.75))
        note_name = BAR_NOTES[bar % 4][step_in_bar // 2]
        hz_val = NOTE_HZ.get(note_name, 440.0) + (0.45 if (step_in_bar // 2) % 2 == 0 else -0.35)
        mot = analysis["per_frame"]["motion_energy"][idx]

        # Draw sleek pixel telemetry overlay on a PIL image
        pil_img = Image.fromarray(styled)
        w, h = pil_img.size
        draw = ImageDraw.Draw(pil_img)

        # Top-left badge: active style & bar/step clock
        draw.rectangle([8, 8, 286, 30], fill=(8, 10, 24))
        draw.rectangle([8, 8, 286, 30], outline=(0, 245, 212), width=1)
        draw.text(
            (14, 13),
            f"STYLE: {style_label} | BAR {bar+1:02d}.{step_in_bar:02d}",
            fill=(0, 245, 212),
        )

        # Bottom telemetry bar: Note + exact Hz + Motion Energy bar
        draw.rectangle([8, h - 32, w - 8, h - 8], fill=(8, 10, 24))
        draw.rectangle([8, h - 32, w - 8, h - 8], outline=(45, 55, 95), width=1)
        draw.text(
            (14, h - 25),
            f"NOTE: {note_name} ({hz_val:.2f} Hz)  MOTION:",
            fill=(255, 214, 102),
        )
        bar_x0 = 215
        bar_w = int(110 * min(1.0, mot * 1.2))
        draw.rectangle([bar_x0, h - 24, bar_x0 + 110, h - 15], outline=(90, 105, 140), width=1)
        if bar_w > 0:
            draw.rectangle([bar_x0 + 1, h - 23, bar_x0 + bar_w, h - 16], fill=(26, 255, 85))

        # Scene-Cut Bass Drop Flash Badge
        if recent_cut is not None:
            age, _ = recent_cut
            # Subtle neon border pulse on the first 4 frames of a scene cut
            if age < 4:
                draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 42, 109), width=3)
            draw.rectangle([w - 255, 8, w - 8, 30], fill=(180, 12, 68))
            draw.rectangle([w - 255, 8, w - 8, 30], outline=(255, 220, 240), width=1)
            draw.text((w - 247, 13), "SCENE CUT >> 808 BASS DROP!", fill=(255, 255, 255))

        # Zoom-In / Inflation or Zoom-Out / Shrinking Pitch-Glide Badge
        if recent_scale is not None:
            _, sdata = recent_scale
            is_up = sdata["type"] == "zoom_in_or_inflate"
            slide_st = sdata["suggested_slide_semitones"]
            badge_col = (12, 135, 95) if is_up else (95, 35, 165)
            label = (
                f"ZOOM/INFLATE >> +{slide_st}st HZ RISE"
                if is_up
                else f"ZOOM-OUT/SHRINK >> {slide_st}st GLIDE"
            )
            draw.rectangle([w - 275, h - 32, w - 8, h - 8], fill=badge_col)
            draw.rectangle([w - 275, h - 32, w - 8, h - 8], outline=(0, 245, 212), width=1)
            draw.text((w - 268, h - 25), label, fill=(255, 255, 255))

        final_u8 = np.asarray(pil_img)
        rendered_frames.append(final_u8)

        if idx in (12, 120, 212, 302, 425, 565):
            keyframe_samples.append(final_u8)

    # Save a 3x2 keyframe summary image to inspect visually
    if keyframe_samples:
        kh, kw, _ = keyframe_samples[0].shape
        grid = Image.new("RGB", (kw * 3, kh * 2), (8, 9, 20))
        for i, kf in enumerate(keyframe_samples[:6]):
            r, c = divmod(i, 3)
            grid.paste(Image.fromarray(kf), (c * kw, r * kh))
        grid.save(PREVIEW_GRID_PNG)

    silent_mp4 = SCRATCH_DIR / "styled_doc_silent.mp4"
    iio.imwrite(silent_mp4, rendered_frames, fps=30.0, codec="libx264")

    # Mux with the 24.0s SoundBit soundtrack using FFmpeg (trimming tail reverb to exact 24.0s video length)
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(silent_mp4),
            "-i",
            str(AUDIO_WAV),
            "-t",
            "24.0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(FINAL_MP4),
        ],
        check=True,
    )
    print(f"Final synced video rendered to {FINAL_MP4} in {time.perf_counter() - t0:.1f}s!")


if __name__ == "__main__":
    main()
