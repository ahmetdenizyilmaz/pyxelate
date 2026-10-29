"""Model Context Protocol (MCP) stdio server for Pyxelate + 12 Reel Visual Styles + Video Motion Analyzer.

Run via:
    python -m pyxelate.mcp_server
"""

import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
import imageio.v3 as iio

from .pyx import Pyx
from .pal import Pal
from .styles import apply_style, list_styles, render_styles_showcase, STYLE_FUNCTIONS
from .motion import analyze_video_motion, sync_video_motion_to_audio
from .bar_fills import detect_video_bar_fills
from .stroke_draws import detect_frames_stroke_draws, render_stroke_draws_proof
from .triggers import (
    DEFAULT_TRIGGER_CONFIG,
    extract_video_sound_triggers,
    load_trigger_catalog,
)


TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "style_list",
        "description": "Lists all 12 built-in Reel visual styles (kurzgesagt_glow, chunky_cartoon, bitwonder_bayer, aloboi_dreamcore, oscilloscope_xy, pc98_cyber, isometric_voxel_bevel, neon_blueprint, gameboy_dmg_lcd, thermal_synthwave, halftone_pop_manga, crt_arcade_1989) and classic hardware palettes.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "image_stylize",
        "description": "Applies one of the 12 visual styles or classic Pyx gradient-aware pixelization to an image and writes the output PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Input image file path."},
                "output_path": {"type": "string", "description": "Output PNG file path."},
                "style": {
                    "type": "string",
                    "description": "Style id from style_list (e.g. 'kurzgesagt_glow', 'chunky_cartoon', 'bitwonder_bayer'). Omit to use classic Pyx BGM.",
                },
                "factor": {
                    "type": "integer",
                    "default": 3,
                    "description": "Downsampling pixel factor (2 to 10).",
                },
                "upscale": {
                    "type": "integer",
                    "default": 3,
                    "description": "Nearest-neighbor upscale factor.",
                },
                "palette": {
                    "type": "integer",
                    "default": 12,
                    "description": "Number of colors when using classic Pyx mode.",
                },
                "dither": {
                    "type": "string",
                    "default": "bayer",
                    "description": "Dithering mode for classic Pyx ('none', 'naive', 'bayer', 'floyd', 'atkinson').",
                },
            },
            "required": ["input_path", "output_path"],
        },
    },
    {
        "name": "image_showcase_12_styles",
        "description": "Renders a 4x3 labeled contact sheet showcasing all 12 built-in visual styles on an input image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Input image file path."},
                "output_path": {"type": "string", "description": "Output contact-sheet PNG path."},
                "tile_width": {"type": "integer", "default": 320},
                "tile_height": {"type": "integer", "default": 180},
            },
            "required": ["input_path", "output_path"],
        },
    },
    {
        "name": "video_analyze_motion",
        "description": "Analyzes a video clip for major scene cuts, camera zoom-ins/zoom-outs, object inflation/shrinking (radial optical flow divergence), and per-frame motion energy for SoundBit audio synchronization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to input video file."},
                "max_frames": {
                    "type": "integer",
                    "description": "Optional max frames to analyze.",
                },
            },
            "required": ["video_path"],
        },
    },
    {
        "name": "video_stylize",
        "description": "Stylizes/pixelizes a video file using one of the 12 styles (or cycling styles on scene cuts) and optionally muxes a synced audio WAV file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_video": {"type": "string", "description": "Input video path."},
                "output_video": {"type": "string", "description": "Output MP4 video path."},
                "style": {
                    "type": "string",
                    "default": "bitwonder_bayer",
                    "description": "Primary style name, or 'cycle_on_cuts' to switch styles on major scene cuts.",
                },
                "factor": {"type": "integer", "default": 3},
                "audio_wav": {
                    "type": "string",
                    "description": "Optional WAV soundtrack to mux into the output MP4.",
                },
            },
            "required": ["input_video", "output_video"],
        },
    },
    {
        "name": "video_sync_motion_to_audio",
        "description": "Refines video camera zooms, pullbacks, focal tracking, and sub-bass micro-zoom punches in pre-pixel high-res space to lock tightly onto a rendered music WAV track, then applies a visual style (default 'pc98_cyber') and muxes the final MP4.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_video": {"type": "string", "description": "Input high-res video path."},
                "audio_wav": {"type": "string", "description": "Rendered music WAV path to lock zooms onto."},
                "output_video": {"type": "string", "description": "Output synced & stylized MP4 path."},
                "bpm": {"type": "number", "default": 120.0},
                "style": {"type": "string", "default": "pc98_cyber"},
                "factor": {"type": "integer", "default": 3},
                "phrase_zoom_strength": {"type": "number", "default": 0.16},
                "bass_punch_strength": {"type": "number", "default": 0.062},
            },
            "required": ["input_video", "audio_wav", "output_video"],
        },
    },
    {
        "name": "video_detect_bar_fills",
        "description": "Detects 1D horizontal (`horizontal_ltr`, `horizontal_rtl`) and vertical (`vertical_btt`, `vertical_ttb`) progress bars, energy/dose meters, gauges, capacitor/column fills, and stacked multi-bar arrays in any video, returning frame-accurate `fill_curve` trajectories in [0.0, 1.0] for 1:1 song-part synchronization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Input video file path."},
                "max_frames": {"type": "integer", "description": "Optional max frames."},
                "min_duration_s": {
                    "type": "number",
                    "default": 0.65,
                    "description": "Minimum duration in seconds for a valid progressive bar fill/drain.",
                },
            },
            "required": ["video_path"],
        },
    },
    {
        "name": "video_detect_stroke_draws",
        "description": "Standalone modular detector for progressive indicator drawings (`circle_draw`, `contour_outline_draw` around objects, and `callout_line_draw` pointer lines). Returns exact drawing `start_s`, `finish_s`, `is_closed_loop`, `sweep_direction`, `draw_curve` in [0.0, 1.0], and `song_part_mapping` so scripts can selectively map chosen drawings to song parts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Input video file path."},
                "max_frames": {"type": "integer", "description": "Optional max frames."},
                "min_duration_s": {
                    "type": "number",
                    "default": 0.28,
                    "description": "Minimum progressive drawing duration in seconds.",
                },
                "allowed_shapes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["circle_draw", "contour_outline_draw", "callout_line_draw"],
                    },
                    "description": "Optional subset of indicator shape types to detect.",
                },
                "proof_image_path": {
                    "type": "string",
                    "description": "Optional path to save a START -> DRAWING -> FINISH + draw_curve visual verification sheet.",
                },
            },
            "required": ["video_path"],
        },
    },
    {
        "name": "video_extract_sound_triggers",
        "description": "Extracts configurable movement-to-sound triggers (horizontal_bar_fill, vertical_bar_fill, configurable sparse camera_zoom_in/out, hud_text_typing, state_confirm_chime, sustained_acceleration_riser, energy_emission_burst, mechanical_step_train, micro_particle_collision) and automatic musical arrangement regimes from any video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Input video file path."},
                "max_frames": {"type": "integer", "description": "Optional max frames."},
                "include_catalog": {"type": "boolean", "default": False},
                "zoom_mode": {
                    "type": "string",
                    "enum": ["off", "major_only", "probabilistic", "musical_subtle", "all"],
                    "default": "major_only",
                    "description": "Controls how often camera zooms trigger sweeps. Default 'major_only' avoids triggering on every minor zoom.",
                },
                "zoom_cooldown_s": {
                    "type": "number",
                    "default": 14.0,
                    "description": "Minimum cooldown in seconds between zoom triggers.",
                },
                "zoom_min_sigma": {
                    "type": "number",
                    "default": 2.25,
                    "description": "Minimum radial divergence sigma threshold for 'major_only' zooms.",
                },
                "zoom_include_whoosh": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether zoom triggers also layer an air whoosh burst.",
                },
                "enable_bar_fills": {
                    "type": "boolean",
                    "default": True,
                    "description": "Detect horizontal & vertical bar fills as exact curve-locked song parts.",
                },
                "enable_stroke_draws": {
                    "type": "boolean",
                    "default": False,
                    "description": "Opt-in flag (default False): include circle/contour/callout drawings from `stroke_draws.py`.",
                },
                "config": {
                    "type": "object",
                    "description": "Optional full trigger config dictionary overriding DEFAULT_TRIGGER_CONFIG.",
                },
            },
            "required": ["video_path"],
        },
    },
]


def _handle_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if name == "style_list":
        return {
            "styles": list_styles(),
            "classic_palettes": Pal.list(),
            "count": len(list_styles()),
        }

    if name == "image_stylize":
        in_path = Path(args["input_path"])
        out_path = Path(args["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = np.asarray(Image.open(in_path).convert("RGB"))
        style = args.get("style")
        factor = int(args.get("factor", 3))
        upscale = int(args.get("upscale", factor))
        if style:
            out_arr = apply_style(img, style=style, factor=factor, upscale=upscale)
        else:
            pyx = Pyx(
                factor=factor,
                upscale=upscale,
                palette=int(args.get("palette", 12)),
                dither=args.get("dither", "bayer"),
            )
            out_arr = pyx.fit_transform(img)
        Image.fromarray(out_arr).save(out_path)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "output_path": str(out_path),
            "style": style or "classic_pyx",
            "shape": list(out_arr.shape),
            "elapsed_ms": elapsed_ms,
        }

    if name == "image_showcase_12_styles":
        in_path = Path(args["input_path"])
        out_path = Path(args["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = np.asarray(Image.open(in_path).convert("RGB"))
        tw = int(args.get("tile_width", 320))
        th = int(args.get("tile_height", 180))
        render_styles_showcase(img, str(out_path), tile_width=tw, tile_height=th)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "output_path": str(out_path),
            "styles_rendered": len(STYLE_FUNCTIONS),
            "elapsed_ms": elapsed_ms,
        }

    if name == "video_analyze_motion":
        return analyze_video_motion(
            args["video_path"], max_frames=args.get("max_frames")
        )

    if name == "video_stylize":
        in_vid = args["input_video"]
        out_vid = args["output_video"]
        style = args.get("style", "bitwonder_bayer")
        factor = int(args.get("factor", 3))
        audio_wav = args.get("audio_wav")

        frames = [f for f in iio.imiter(in_vid)]
        try:
            fps = float(iio.immeta(in_vid).get("fps", 30.0))
        except Exception:
            fps = 30.0

        styled_frames = []
        for idx, frame in enumerate(frames):
            styled_frames.append(
                apply_style(frame, style=style, factor=factor, upscale=factor, frame_index=idx)
            )

        Path(out_vid).parent.mkdir(parents=True, exist_ok=True)
        if audio_wav:
            import shutil
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            tmp_vid = str(Path(out_vid).with_suffix(".silent.mp4"))
            iio.imwrite(tmp_vid, styled_frames, fps=fps, codec="libx264")
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
                    out_vid,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            Path(tmp_vid).unlink(missing_ok=True)
        else:
            iio.imwrite(out_vid, styled_frames, fps=fps, codec="libx264")

        return {
            "output_video": out_vid,
            "frames": len(styled_frames),
            "fps": fps,
            "style": style,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    if name == "video_sync_motion_to_audio":
        res = sync_video_motion_to_audio(
            input_video=args["input_video"],
            audio_wav=args["audio_wav"],
            output_video=args["output_video"],
            bpm=float(args.get("bpm", 120.0)),
            style=args.get("style", "pc98_cyber"),
            factor=int(args.get("factor", 3)),
            phrase_zoom_strength=float(args.get("phrase_zoom_strength", 0.16)),
            bass_punch_strength=float(args.get("bass_punch_strength", 0.062)),
        )
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return res

    if name == "video_detect_bar_fills":
        res = detect_video_bar_fills(
            args["video_path"],
            max_frames=args.get("max_frames"),
            min_duration_s=float(args.get("min_duration_s", 0.65)),
        )
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return res

    if name == "video_detect_stroke_draws":
        vid_p = args["video_path"]
        max_f = args.get("max_frames")
        fps = 30.0
        try:
            fps = float(iio.immeta(vid_p).get("fps", 30.0))
        except Exception:
            fps = 30.0
        frames = []
        for frame in iio.imiter(vid_p):
            frames.append(frame)
            if max_f is not None and len(frames) >= int(max_f):
                break
        res = detect_frames_stroke_draws(
            frames,
            fps=fps,
            min_duration_s=float(args.get("min_duration_s", 0.28)),
            allowed_shapes=args.get("allowed_shapes"),
        )
        res["video_path"] = vid_p
        if args.get("proof_image_path"):
            res["proof_image_path"] = render_stroke_draws_proof(
                frames, res, str(args["proof_image_path"])
            )
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return res

    if name == "video_extract_sound_triggers":
        cfg = dict(DEFAULT_TRIGGER_CONFIG)
        if isinstance(args.get("config"), dict):
            cfg.update(args["config"])
        for key in (
            "zoom_mode",
            "zoom_cooldown_s",
            "zoom_min_sigma",
            "zoom_include_whoosh",
            "enable_bar_fills",
            "enable_stroke_draws",
        ):
            if key in args and args[key] is not None:
                cfg[key] = args[key]
        res = extract_video_sound_triggers(
            args["video_path"],
            max_frames=args.get("max_frames"),
            config=cfg,
        )
        if args.get("include_catalog"):
            res["catalog"] = load_trigger_catalog()
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return res

    raise ValueError(f"Unknown tool: {name}")


def run_stdio_server() -> None:
    """Runs JSON-RPC 2.0 MCP server over stdin/stdout."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        if req_id is None:
            # Notification (e.g. notifications/initialized)
            continue

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "pyxelate-mcp", "version": "2.2.0"},
                },
            }
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS_SCHEMA},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments") or {}
            try:
                data = _handle_tool_call(tool_name, tool_args)
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
