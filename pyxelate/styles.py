"""12 Signature Visual Styles for Pyxelate Reels & Documentary Stylization.

Includes:
1.  `kurzgesagt_glow`       - Kurzgesagt "In a Nutshell" flat-vector regions, cosmic indigo shadows,
                              bioluminescent neon cyan/green/magenta rim lighting, and outer bloom.
2.  `chunky_cartoon`        - Bold Cel-Shaded Chunky Cartoon with Kuwahara brush blocks, saturated
                              6-band posterization, and multi-scale black comic ink outlines.
3.  `bitwonder_bayer`       - @BITWONDER 16-bit keygen style: Sobel gradient pixelization, 4x4 Bayer
                              crosshatch dithering in OKLab space, and 1px dark sprite outlines.
4.  `aloboi_dreamcore`      - Warm analog synthwave / VHS dreamcore with golden-magenta split-toning,
                              ordered pixel dither, RGB chromatic aberration, and CRT scanlines.
5.  `oscilloscope_xy`       - Jerobeam Fenderson / P31 Phosphor (#00ff66) vector oscilloscope beam
                              with harmonic XY wave modulation and calibrated dark CRT graticule.
6.  `pc98_cyber`            - NEC PC-9801 16-color Japanese Cyber-Anime ordered dither with neon
                              cyan/magenta/violet ramps and crisp 1px dark line reinforcement.
7.  `isometric_voxel_bevel` - 3D Chunky Voxel / Micro-Tile blocks with top-left specular bevel
                              highlights, bottom-right shadow bevels, and recessed tile grooves.
8.  `neon_blueprint`        - Cybernetic Hologram / Architectural Blueprint with glowing cyan/amber
                              wireframe contours, technical grid, and corner telemetry markers.
9.  `gameboy_dmg_lcd`       - Authentic Nintendo DMG-01 4-shade olive dot-matrix LCD with sub-pixel
                              matrix grid shadow.
10. `thermal_synthwave`     - Predator FLIR / Infrared Heat-Vision with 16-stop neon plasma palette,
                              Bayer heat-dither, and isothermal contour rings.
11. `halftone_pop_manga`    - Spider-Verse CMYK angled halftone dot matrices + bold black manga ink
                              contours and shadow cross-hatching.
12. `crt_arcade_1989`       - 1989 Trinitron 16-bit Arcade CRT with RGB phosphor aperture grille,
                              scanline darkening, and glass halation bloom.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter
from PIL import Image, ImageDraw, ImageFont

from .fast_ops import (
    BAYER_4X4,
    BAYER_8X8,
    resize_float,
    rgb2hsv_fast,
    hsv2rgb_fast,
    oklab_palette_labels,
    equalize_adapthist_fast,
    sobel_downsample_numba,
    kuwahara_filter_numba,
)


STYLE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "kurzgesagt_glow",
        "name": "Kurzgesagt Cosmic Vector Glow",
        "category": "cartoon_vector",
        "description": "Flat-vector Kuwahara regions mapped to Kurzgesagt's signature cosmic navy/indigo, bioluminescent neon green/cyan, ultraviolet purple, and warm coral/gold with rim highlights and multi-stage outer glow.",
    },
    {
        "id": "chunky_cartoon",
        "name": "Chunky Cel-Shaded Toon",
        "category": "cartoon_vector",
        "description": "Bold chunky painterly color blocks with high-saturation 6-band cel shading and thick adaptive black comic-book ink contours.",
    },
    {
        "id": "bitwonder_bayer",
        "name": "BitWonder 16-Bit Bayer Sprite",
        "category": "pixel_art",
        "description": "Gradient-aware Sobel downsampling, 4x4 ordered Bayer crosshatch dithering in perceptual OKLab space, and 1px dark sprite outlines.",
    },
    {
        "id": "aloboi_dreamcore",
        "name": "Aloboi Analog VHS Dreamcore",
        "category": "retro_synth",
        "description": "Warm golden-hour & magenta-indigo analog synthwave grading with pixel dither, horizontal RGB chromatic drift, and soft VHS scanlines.",
    },
    {
        "id": "oscilloscope_xy",
        "name": "Oscilloscope XY Vector Phosphor",
        "category": "retro_synth",
        "description": "Glowing #00ff66 P31 phosphor vector contours and harmonic scanline modulation over a calibrated dark CRT oscilloscope graticule.",
    },
    {
        "id": "pc98_cyber",
        "name": "NEC PC-9801 Cyber Anime",
        "category": "pixel_art",
        "description": "Authentic 16-color NEC PC-9801 cyberpunk palette with ordered 4x4 Bayer dithering and crisp anime line-art reinforcement.",
    },
    {
        "id": "isometric_voxel_bevel",
        "name": "3D Chunky Voxel Bevel",
        "category": "pixel_art",
        "description": "Chunky quantized macro-pixels rendered as 3D beveled tiles with top-left specular highlights and bottom-right shadow grooves.",
    },
    {
        "id": "neon_blueprint",
        "name": "Cybernetic Neon Blueprint",
        "category": "sci_fi",
        "description": "Prussian-blue technical schematic with glowing cyan structural edges, amber high-energy focal highlights, and blueprint grid.",
    },
    {
        "id": "gameboy_dmg_lcd",
        "name": "Game Boy DMG-01 Dot-Matrix LCD",
        "category": "pixel_art",
        "description": "Classic 4-shade olive green Game Boy palette with ordered Bayer dithering and authentic sub-pixel LCD matrix grid lines.",
    },
    {
        "id": "thermal_synthwave",
        "name": "Thermal Synthwave FLIR",
        "category": "sci_fi",
        "description": "Infrared heat-vision mapping luminance and motion energy to ultraviolet, neon magenta, laser orange, and white-hot cores with Bayer dither.",
    },
    {
        "id": "halftone_pop_manga",
        "name": "Spider-Verse Halftone Pop Manga",
        "category": "cartoon_vector",
        "description": "Vibrant posterized pop-art color planes overlaid with angled halftone screen-tone dots and heavy black comic ink outlines.",
    },
    {
        "id": "crt_arcade_1989",
        "name": "1989 Trinitron Arcade CRT",
        "category": "retro_synth",
        "description": "32-color 16-bit arcade pixel art with RGB aperture-grille phosphor subpixels, scanline darkening, and warm glass halation bloom.",
    },
]


def _hex_palette(hex_list: List[str]) -> np.ndarray:
    return (
        np.array(
            [[int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] for h in hex_list],
            dtype=np.float64,
        )
        / 255.0
    )


# Curated Palettes
KURZGESAGT_PALETTE = _hex_palette(
    [
        "#050517",  # Deep cosmic void
        "#0c0f30",  # Space navy
        "#181d52",  # Indigo shadow
        "#2b2d7c",  # Twilight blue-violet
        "#4c1d95",  # Deep ultraviolet
        "#7b2cbf",  # Vibrant Kurzgesagt purple
        "#b5179e",  # Neon magenta-violet
        "#f72585",  # Hot coral pink
        "#0055ff",  # Electric royal blue
        "#00bbf9",  # Sky cyan
        "#00f5d4",  # Bioluminescent aqua-cyan
        "#1aff55",  # Kurzgesagt radioactive green
        "#80ff72",  # Luminous mint-lime highlight
        "#ff9e00",  # Warm golden amber
        "#ff5400",  # Vibrant orange
        "#ffee99",  # Starlight warm cream
        "#ffffff",  # Pure starlight white
    ]
)

BITWONDER_PALETTE = _hex_palette(
    [
        "#080914",
        "#14182e",
        "#262b44",
        "#3a4466",
        "#5a6988",
        "#8b9bb4",
        "#c0cbdc",
        "#ffffff",
        "#ff0044",
        "#ff5500",
        "#ffaa00",
        "#ffee22",
        "#63c74d",
        "#00e436",
        "#00f5d4",
        "#0099db",
        "#29adff",
        "#1d2b53",
        "#7e2553",
        "#ab5236",
        "#ff77a8",
        "#ffccaa",
        "#83769c",
        "#a8e72e",
    ]
)

PC98_PALETTE = _hex_palette(
    [
        "#080614",
        "#191233",
        "#2e1f5e",
        "#4d2d8c",
        "#7b2cbf",
        "#c77dff",
        "#ff2a85",
        "#ff70a6",
        "#003566",
        "#0077b6",
        "#00b4d8",
        "#00f5d4",
        "#ff9e00",
        "#ffd166",
        "#f8edeb",
        "#ffffff",
    ]
)

DMG_PALETTE = _hex_palette(["#0f380f", "#306230", "#8bac0f", "#9bbc0f"])

THERMAL_PALETTE = _hex_palette(
    [
        "#040210",
        "#12062c",
        "#2a0944",
        "#4b0d63",
        "#7209b7",
        "#a01a7d",
        "#d90429",
        "#f72585",
        "#ff4800",
        "#ff7900",
        "#ffaa00",
        "#ffd000",
        "#ffee55",
        "#ffffaa",
        "#ffffff",
    ]
)


def _edge_magnitude(rgb: np.ndarray) -> np.ndarray:
    """Fast perceptual luminance + chroma Sobel gradient magnitude in [0, 1]."""
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gy, gx = np.gradient(luma)
    cy, cx = np.gradient(rgb[:, :, 0] - rgb[:, :, 1])
    mag = np.sqrt(gx * gx + gy * gy + 0.35 * (cx * cx + cy * cy))
    return np.clip(mag * 4.2, 0.0, 1.0)


def _quantize_with_bayer(
    rgb: np.ndarray, palette: np.ndarray, spread: float = 0.08, use_8x8: bool = False
) -> np.ndarray:
    """Quantizes RGB image [0..1] to `palette` with ordered Bayer dithering in OKLab space."""
    h, w, _ = rgb.shape
    mat = BAYER_8X8 if use_8x8 else BAYER_4X4
    mh, mw = mat.shape
    tiled = np.tile(mat, (h // mh + 1, w // mw + 1))[:h, :w, None]
    dithered = np.clip(rgb + tiled * spread, 0.0, 1.0)
    labels = oklab_palette_labels(dithered, palette)
    return palette[labels].reshape(h, w, 3)


def _lift_tones(rgb: np.ndarray, gamma: float = 0.82, highlight_compress: float = 0.92) -> np.ndarray:
    """Lifts crushed shadows and compresses blown highlights so all midtone details stay vivid."""
    lifted = np.power(np.clip(rgb, 0.0, 1.0), gamma)
    # Softly compress top 15% of highlights so skies/backgrounds retain rich color instead of white clipping
    compressed = np.where(
        lifted > 0.78,
        0.78 + (lifted - 0.78) * highlight_compress * 0.65,
        lifted,
    )
    return np.clip(compressed, 0.0, 1.0)


def _style_kurzgesagt_glow(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Kurzgesagt 'In a Nutshell' / 'THE LAST HUMAN' flat-vector + bioluminescent neon bloom style."""
    h, w, _ = rgb.shape
    work_h, work_w = max(56, h // max(1, factor)), max(56, w // max(1, factor))
    small = _lift_tones(resize_float(rgb, work_h, work_w, anti_aliasing=True), gamma=0.78)

    # 1. Multi-pass Kuwahara flat-vector smoothing (creates rounded, flat illustration shapes)
    flat = kuwahara_filter_numba(small, radius=3)
    flat = kuwahara_filter_numba(flat, radius=2)

    # 2. Remap into Kurzgesagt's vibrant cosmic color harmony (no crushed blacks or blown whites!)
    hsv = rgb2hsv_fast(flat)
    luma = 0.299 * flat[:, :, 0] + 0.587 * flat[:, :, 1] + 0.114 * flat[:, :, 2]

    # Deepest background (<0.15) -> cosmic indigo; mid-shadows (0.15..0.42) -> vibrant purple/blue/teal
    deep_bg = luma < 0.15
    mid_shadow = (luma >= 0.15) & (luma < 0.42)
    hsv[:, :, 0] = np.where(deep_bg, 0.67, hsv[:, :, 0])
    # Shift low-saturation midtones into Kurzgesagt cyan/purple/emerald/coral based on luminance & hue
    low_sat = hsv[:, :, 1] < 0.28
    synthetic_hue = np.mod(hsv[:, :, 0] * 1.4 + luma * 0.55 + 0.48, 1.0)
    hsv[:, :, 0] = np.where(low_sat & ~deep_bg, synthetic_hue, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(
        deep_bg,
        0.78,
        np.clip(hsv[:, :, 1] * 1.65 + 0.42, 0.52, 0.98),
    )
    # Posterize brightness into 7 vibrant illustration bands in [0.10, 0.94]
    val_mapped = 0.10 + 0.84 * (np.round(luma * 6.0) / 6.0)
    val_mapped = np.where(mid_shadow, np.maximum(val_mapped, 0.34), val_mapped)
    hsv[:, :, 2] = np.clip(val_mapped, 0.08, 0.95)
    vivid = hsv2rgb_fast(hsv)

    # Blend 60% OKLab Kurzgesagt palette + 40% posterized vector ramp
    pal_matched = _quantize_with_bayer(vivid, KURZGESAGT_PALETTE, spread=0.03)
    vector_img = np.clip(0.60 * pal_matched + 0.40 * vivid, 0.0, 1.0)

    # 3. Clean silhouettes + luminous neon green/cyan rim highlights along edges
    edges = _edge_magnitude(flat)
    outer_silhouette = (edges > 0.36).astype(np.float64)
    inner_rim = ((edges > 0.17) & (edges <= 0.36)).astype(np.float64)

    indigo_ink = np.array([0.04, 0.05, 0.16])
    vector_img = vector_img * (1.0 - 0.72 * outer_silhouette[:, :, None]) + indigo_ink * (
        0.72 * outer_silhouette[:, :, None]
    )

    rim_color = np.array([0.08, 0.98, 0.64])
    vector_img = np.clip(vector_img + inner_rim[:, :, None] * rim_color * 0.38, 0.0, 1.0)

    # 4. Upscale
    out_h, out_w = work_h * upscale, work_w * upscale
    up = np.repeat(np.repeat(vector_img, upscale, axis=0), upscale, axis=1)

    # 5. Screen-blend Bioluminescent Bloom (never clips to flat white!)
    up_luma = 0.299 * up[:, :, 0] + 0.587 * up[:, :, 1] + 0.114 * up[:, :, 2]
    up_sat = rgb2hsv_fast(up)[:, :, 1]
    glow_weight = np.clip((up_luma - 0.42) * up_sat * 1.4, 0.0, 1.0)[:, :, None]
    emitter = up * glow_weight
    bloom_tight = gaussian_filter(emitter, sigma=(3.0, 3.0, 0.0))
    bloom_wide = gaussian_filter(emitter, sigma=(9.0, 9.0, 0.0))
    bloom_total = np.clip(0.42 * bloom_tight + 0.32 * bloom_wide, 0.0, 0.65)
    up = 1.0 - (1.0 - up) * (1.0 - bloom_total)

    # 6. Subtle deterministic cosmic starfield in dark background regions
    up_edges = np.repeat(
        np.repeat(gaussian_filter(edges, sigma=1.5), upscale, axis=0), upscale, axis=1
    )
    deep_space = (up_luma < 0.22) & (up_edges < 0.09)
    if np.any(deep_space):
        yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
        star_hash = ((xx * 73856093) ^ (yy * 19349663)) & 0xFFFF
        twinkle = 0.65 + 0.35 * np.sin(frame_index * 0.25 + (star_hash % 31))
        stars = (star_hash < 58) & deep_space
        up[stars] = np.clip(
            up[stars] + np.array([0.75, 0.90, 1.0]) * twinkle[stars, None], 0.0, 1.0
        )

    return resize_float(up, h, w, anti_aliasing=False)


def _style_chunky_cartoon(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Chunky Cel-Shaded Cartoon style with Kuwahara paint blocks and bold comic ink outlines."""
    h, w, _ = rgb.shape
    work_h, work_w = max(56, h // max(1, factor)), max(56, w // max(1, factor))
    small = _lift_tones(
        equalize_adapthist_fast(resize_float(rgb, work_h, work_w, anti_aliasing=True)),
        gamma=0.80,
    )

    # 1. Strong Kuwahara brush-block simplification
    chunky = kuwahara_filter_numba(small, radius=4)
    chunky = kuwahara_filter_numba(chunky, radius=2)

    # 2. High-contrast 6-band HSV cel-shading with lifted shadows so black ink outlines pop!
    hsv = rgb2hsv_fast(chunky)
    hsv[:, :, 0] = np.round(hsv[:, :, 0] * 18.0) / 18.0
    hsv[:, :, 1] = np.clip(np.round(hsv[:, :, 1] * 4.0) / 4.0 * 1.55 + 0.18, 0.22, 0.96)
    hsv[:, :, 2] = np.clip(0.22 + 0.74 * (np.round(hsv[:, :, 2] * 5.0) / 5.0), 0.22, 0.96)
    cel = hsv2rgb_fast(hsv)

    # 3. Bold adaptive black ink outlines (Difference of Gaussians + Sobel)
    luma = 0.299 * chunky[:, :, 0] + 0.587 * chunky[:, :, 1] + 0.114 * chunky[:, :, 2]
    dog = gaussian_filter(luma, sigma=0.7) - gaussian_filter(luma, sigma=2.0)
    edges = _edge_magnitude(chunky)
    ink_mask = (edges > 0.25) | (dog < -0.028)

    ink_color = np.array([0.04, 0.03, 0.08])
    cel[ink_mask] = ink_color

    up = np.repeat(np.repeat(cel, upscale, axis=0), upscale, axis=1)
    return resize_float(up, h, w, anti_aliasing=False)


def _style_bitwonder_bayer(
    rgb: np.ndarray, factor: int = 4, upscale: int = 4, frame_index: int = 0
) -> np.ndarray:
    """@BITWONDER 16-bit keygen pixel art: Sobel downsampling + 4x4 Bayer OKLab dither + 1px sprite outlines."""
    h, w, _ = rgb.shape
    tile = max(2, int(factor))
    pad_h = (h // tile) * tile
    pad_w = (w // tile) * tile
    cropped = _lift_tones(
        equalize_adapthist_fast(resize_float(rgb, pad_h, pad_w, anti_aliasing=True)),
        gamma=0.84,
    )

    # 1. Fast Numba Sobel-gradient-weighted downsampling
    down = sobel_downsample_numba(cropped, tile)

    # 2. Boost saturation for punchy 16-bit keygen sprites
    hsv = rgb2hsv_fast(down)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.42 + 0.10, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(0.12 + hsv[:, :, 2] * 0.86, 0.08, 0.98)
    vivid = hsv2rgb_fast(hsv)

    # 3. Ordered 4x4 Bayer crosshatch dithering in perceptual OKLab space
    pix = _quantize_with_bayer(vivid, BITWONDER_PALETTE, spread=0.085, use_8x8=False)

    # 4. Crisp 1px dark sprite outlines around strong depth/luminance discontinuities
    edges = _edge_magnitude(down)
    outline = edges > 0.36
    pix[outline] = np.array([0.031, 0.035, 0.078])

    up = np.repeat(np.repeat(pix, tile, axis=0), tile, axis=1)
    return resize_float(up, h, w, anti_aliasing=False)


def _style_aloboi_dreamcore(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Aloboi-inspired warm analog synthwave / VHS dreamcore pixel style."""
    h, w, _ = rgb.shape
    tile = max(2, int(factor))
    small = resize_float(rgb, max(48, h // tile), max(48, w // tile), anti_aliasing=True)

    # 1. Golden-hour & twilight-magenta split toning
    luma = (0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2])[:, :, None]
    shadow_tint = np.array([0.16, 0.06, 0.34])  # Deep analog violet
    mid_tint = np.array([0.92, 0.32, 0.58])     # Warm rose-magenta
    high_tint = np.array([1.00, 0.86, 0.48])    # Golden sunset
    graded = (
        small * 0.62
        + (1.0 - luma) * shadow_tint * 0.38
        + (1.0 - np.abs(luma - 0.5) * 2.0) * mid_tint * 0.22
        + (luma * luma) * high_tint * 0.36
    )
    graded = np.clip(graded, 0.0, 1.0)

    # 2. Ordered 8x8 Bayer dithering onto 16-level warm DAC
    mat = BAYER_8X8
    tiled = np.tile(mat, (small.shape[0] // 8 + 1, small.shape[1] // 8 + 1))[
        : small.shape[0], : small.shape[1], None
    ]
    quant = np.clip(np.round((graded + tiled * 0.055) * 9.0) / 9.0, 0.0, 1.0)

    up = resize_float(quant, h, w, anti_aliasing=False)

    # 3. Analog RGB horizontal chromatic aberration + VHS scanlines + halation bloom
    shift = max(1, w // 240)
    aberrated = np.empty_like(up)
    aberrated[:, :, 0] = np.roll(up[:, :, 0], shift, axis=1)
    aberrated[:, :, 1] = up[:, :, 1]
    aberrated[:, :, 2] = np.roll(up[:, :, 2], -shift, axis=1)

    scanline = (0.88 + 0.12 * np.sin((np.arange(h) + frame_index * 0.5) * np.pi))[:, None, None]
    bloom = gaussian_filter(np.maximum(aberrated - 0.55, 0.0), sigma=(6.0, 6.0, 0.0))
    return np.clip(aberrated * scanline + bloom * 0.65, 0.0, 1.0)


def _style_oscilloscope_xy(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Jerobeam Fenderson / Phosphor Green (#00ff66) XY Vector Oscilloscope style."""
    h, w, _ = rgb.shape
    small = resize_float(rgb, max(64, h // 2), max(64, w // 2), anti_aliasing=True)
    edges = _edge_magnitude(small)
    luma = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]

    # Harmonic scanline wave modulation (creates authentic oscilloscope waveform traces)
    sh, sw = small.shape[:2]
    yy, xx = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    wave_trace = np.sin(yy * 0.85 - luma * 9.5 + frame_index * 0.35) > (0.92 - 0.45 * luma)
    beam = np.clip(edges * 1.45 + wave_trace.astype(np.float64) * 0.55 * luma, 0.0, 1.0)
    beam_full = resize_float(beam, h, w, anti_aliasing=False)

    # Calibrated dark CRT graticule background
    bg = np.zeros((h, w, 3), dtype=np.float64)
    bg[:, :] = np.array([0.012, 0.038, 0.022])
    grid_y = (np.arange(h) % max(16, h // 12) == 0)[:, None]
    grid_x = (np.arange(w) % max(16, w // 12) == 0)[None, :]
    bg[grid_y | grid_x] = np.array([0.035, 0.11, 0.06])

    # Phosphor P31 green core + emerald outer bloom
    glow_tight = gaussian_filter(beam_full, sigma=1.8)
    glow_wide = gaussian_filter(beam_full, sigma=7.0)
    phosphor = (
        bg
        + beam_full[:, :, None] * np.array([0.55, 1.0, 0.72])
        + glow_tight[:, :, None] * np.array([0.05, 0.85, 0.38]) * 0.85
        + glow_wide[:, :, None] * np.array([0.0, 0.55, 0.22]) * 0.75
    )
    return np.clip(phosphor, 0.0, 1.0)


def _style_pc98_cyber(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """NEC PC-9801 16-Color Japanese Cyber-Anime ordered dither style."""
    h, w, _ = rgb.shape
    tile = max(2, int(factor))
    small = _lift_tones(
        equalize_adapthist_fast(resize_float(rgb, max(48, h // tile), max(48, w // tile), anti_aliasing=True)),
        gamma=0.82,
    )

    # Slight cyan/magenta anime tint
    tinted = np.clip(small * np.array([1.05, 0.96, 1.12]), 0.0, 1.0)
    pix = _quantize_with_bayer(tinted, PC98_PALETTE, spread=0.095, use_8x8=False)

    # Reinforce dark anime linework
    edges = _edge_magnitude(small)
    pix[edges > 0.34] = PC98_PALETTE[0]

    up = np.repeat(np.repeat(pix, tile, axis=0), tile, axis=1)
    return resize_float(up, h, w, anti_aliasing=False)


def _style_isometric_voxel_bevel(
    rgb: np.ndarray, factor: int = 6, upscale: int = 6, frame_index: int = 0
) -> np.ndarray:
    """3D Chunky Voxel / Micro-Tile Bevel style with specular top-left & shadow bottom-right edges."""
    h, w, _ = rgb.shape
    tile = max(5, int(factor) * 2)
    gh, gw = max(24, h // tile), max(24, w // tile)
    small = resize_float(rgb, gh, gw, anti_aliasing=True)
    quant = _quantize_with_bayer(small, BITWONDER_PALETTE, spread=0.04)

    up = np.repeat(np.repeat(quant, tile, axis=0), tile, axis=1)
    uh, uw, _ = up.shape
    yy = np.arange(uh)[:, None] % tile
    xx = np.arange(uw)[None, :] % tile

    # Top/left 1px specular bevel highlight, bottom/right 1px shadow bevel, inner stud/groove
    top_left = (yy == 0) | (xx == 0)
    bot_right = (yy == tile - 1) | (xx == tile - 1)
    inner_highlight = (yy == 1) & (xx >= 1) & (xx < tile - 1)

    up[top_left] = np.clip(up[top_left] * 1.28 + 0.08, 0.0, 1.0)
    up[inner_highlight] = np.clip(up[inner_highlight] * 1.12 + 0.03, 0.0, 1.0)
    up[bot_right] = up[bot_right] * 0.58
    return resize_float(up, h, w, anti_aliasing=False)


def _style_neon_blueprint(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Cybernetic Hologram / Architectural Blueprint style."""
    h, w, _ = rgb.shape
    tile = max(2, int(factor))
    small = resize_float(rgb, max(48, h // tile), max(48, w // tile), anti_aliasing=True)
    luma = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    edges = _edge_magnitude(small)

    # Posterized Prussian blueprint base
    q_luma = np.round(luma * 5.0) / 5.0
    base = (
        np.array([0.02, 0.07, 0.18])[None, None, :] * (1.0 - q_luma[:, :, None])
        + np.array([0.06, 0.28, 0.58])[None, None, :] * q_luma[:, :, None]
    )

    # Glowing cyan primary contours + amber focal highlights
    cyan_wire = np.clip((edges - 0.14) * 2.8, 0.0, 1.0)[:, :, None] * np.array([0.0, 0.96, 0.88])
    amber_hot = ((edges > 0.38) & (luma > 0.55)).astype(np.float64)[:, :, None] * np.array(
        [1.0, 0.75, 0.08]
    )
    combined = np.clip(base + cyan_wire + amber_hot, 0.0, 1.0)

    up = resize_float(combined, h, w, anti_aliasing=False)
    # Technical blueprint grid overlay
    gy = (np.arange(h) % 24 == 0)[:, None]
    gx = (np.arange(w) % 24 == 0)[None, :]
    up[gy | gx] = np.clip(up[gy | gx] + np.array([0.0, 0.14, 0.22]), 0.0, 1.0)
    return up


def _style_gameboy_dmg_lcd(
    rgb: np.ndarray, factor: int = 4, upscale: int = 4, frame_index: int = 0
) -> np.ndarray:
    """Authentic Nintendo DMG-01 4-shade olive dot-matrix LCD with subpixel grid."""
    h, w, _ = rgb.shape
    tile = max(3, int(factor))
    small = resize_float(rgb, max(40, h // tile), max(40, w // tile), anti_aliasing=True)
    luma = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    # Local contrast boost so all 4 DMG shades are used effectively
    luma = equalize_adapthist_fast(np.repeat(luma[:, :, None], 3, axis=2))[:, :, 0]

    sh, sw = luma.shape
    tiled_bayer = np.tile(BAYER_4X4, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw]
    idx = np.clip(np.floor((luma + tiled_bayer * 0.22) * 4.0).astype(np.int32), 0, 3)
    dmg = DMG_PALETTE[idx]

    up = np.repeat(np.repeat(dmg, tile, axis=0), tile, axis=1)
    uh, uw, _ = up.shape
    lcd_grid = ((np.arange(uh)[:, None] % tile) == tile - 1) | (
        (np.arange(uw)[None, :] % tile) == tile - 1
    )
    up[lcd_grid] = up[lcd_grid] * 0.82
    return resize_float(up, h, w, anti_aliasing=False)


def _style_thermal_synthwave(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Predator FLIR / Neon Synthwave Thermal heat-map pixel style."""
    h, w, _ = rgb.shape
    tile = max(2, int(factor))
    small = resize_float(rgb, max(48, h // tile), max(48, w // tile), anti_aliasing=True)
    luma = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    edges = _edge_magnitude(small)
    heat = np.clip(luma * 0.82 + edges * 0.32, 0.0, 1.0)

    sh, sw = heat.shape
    tiled_bayer = np.tile(BAYER_4X4, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw]
    n_stops = len(THERMAL_PALETTE)
    idx = np.clip(
        np.floor((heat + tiled_bayer * 0.08) * n_stops).astype(np.int32), 0, n_stops - 1
    )
    therm = THERMAL_PALETTE[idx]

    up = np.repeat(np.repeat(therm, tile, axis=0), tile, axis=1)
    bloom = gaussian_filter(np.maximum(up - 0.65, 0.0), sigma=(4.0, 4.0, 0.0))
    return resize_float(np.clip(up + bloom * 0.55, 0.0, 1.0), h, w, anti_aliasing=False)


def _style_halftone_pop_manga(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """Spider-Verse Pop-Art CMYK Halftone Dots + Bold Manga Ink Edges."""
    h, w, _ = rgb.shape
    smooth = kuwahara_filter_numba(resize_float(rgb, max(64, h // 2), max(64, w // 2)), radius=2)
    base = resize_float(smooth, h, w, anti_aliasing=True)

    # Boosted pop-art posterization
    hsv = rgb2hsv_fast(base)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.55, 0.0, 1.0)
    hsv[:, :, 2] = np.round(hsv[:, :, 2] * 5.0) / 5.0
    pop = hsv2rgb_fast(hsv)

    # Angled sinusoidal halftone screen pattern
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    angle = 0.52
    u = xx * np.cos(angle) - yy * np.sin(angle)
    v = xx * np.sin(angle) + yy * np.cos(angle)
    dot_wave = 0.5 + 0.25 * (np.cos(u * 0.95) + np.cos(v * 0.95))

    luma = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    shadow_dots = (luma < 0.65) & (dot_wave > (luma + 0.18))
    pop[shadow_dots] *= 0.52

    # Heavy black comic ink outlines
    edges = resize_float(_edge_magnitude(smooth), h, w, anti_aliasing=False)
    pop[edges > 0.26] = np.array([0.04, 0.03, 0.06])
    return np.clip(pop, 0.0, 1.0)


def _style_crt_arcade_1989(
    rgb: np.ndarray, factor: int = 3, upscale: int = 3, frame_index: int = 0
) -> np.ndarray:
    """1989 Trinitron 16-bit Arcade CRT with RGB phosphor aperture grille & scanline bloom."""
    pix = _style_bitwonder_bayer(rgb, factor=factor, upscale=upscale, frame_index=frame_index)
    h, w, _ = pix.shape

    # RGB phosphor triad mask along X + scanlines along Y
    mask = np.ones((h, w, 3), dtype=np.float64)
    x_mod = np.arange(w) % 3
    mask[:, x_mod == 0, 1:] *= 0.78
    mask[:, x_mod == 1, ::2] *= 0.78
    mask[:, x_mod == 2, :2] *= 0.78
    scan = (0.78 + 0.22 * (np.arange(h) % 2))[:, None, None]

    crt = pix * mask * scan
    bloom = gaussian_filter(pix, sigma=(3.5, 3.5, 0.0))
    return np.clip(crt * 0.85 + bloom * 0.38, 0.0, 1.0)


STYLE_FUNCTIONS = {
    "kurzgesagt_glow": _style_kurzgesagt_glow,
    "chunky_cartoon": _style_chunky_cartoon,
    "bitwonder_bayer": _style_bitwonder_bayer,
    "aloboi_dreamcore": _style_aloboi_dreamcore,
    "oscilloscope_xy": _style_oscilloscope_xy,
    "pc98_cyber": _style_pc98_cyber,
    "isometric_voxel_bevel": _style_isometric_voxel_bevel,
    "neon_blueprint": _style_neon_blueprint,
    "gameboy_dmg_lcd": _style_gameboy_dmg_lcd,
    "thermal_synthwave": _style_thermal_synthwave,
    "halftone_pop_manga": _style_halftone_pop_manga,
    "crt_arcade_1989": _style_crt_arcade_1989,
}


def list_styles() -> List[Dict[str, Any]]:
    """Returns metadata for all 12 built-in visual styles."""
    return STYLE_CATALOG


def apply_style(
    image: np.ndarray,
    style: str = "kurzgesagt_glow",
    factor: int = 3,
    upscale: int = 3,
    frame_index: int = 0,
) -> np.ndarray:
    """Applies one of the 12 signature styles to an RGB uint8 or float image, returning uint8 [0..255]."""
    if style not in STYLE_FUNCTIONS:
        valid = ", ".join(STYLE_FUNCTIONS.keys())
        raise ValueError(f"Unknown style '{style}'. Choose from: {valid}")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    rgb = arr[:, :, :3].astype(np.float64)
    if arr.dtype.kind != "f" or rgb.max() > 1.0:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    styled = STYLE_FUNCTIONS[style](
        rgb, factor=max(1, int(factor)), upscale=max(1, int(upscale)), frame_index=int(frame_index)
    )
    return (np.clip(styled, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def render_styles_showcase(
    image: np.ndarray, output_path: str, tile_width: int = 320, tile_height: int = 180
) -> str:
    """Renders a 4x3 labeled HD showcase grid of all 12 styles on `image` and saves to `output_path`."""
    cols, rows = 4, 3
    header_h = 28
    cell_w, cell_h = tile_width, tile_height + header_h
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (8, 9, 20))
    draw = ImageDraw.Draw(canvas)

    thumb_src = resize_float(image, tile_height, tile_width, anti_aliasing=True)
    for idx, meta in enumerate(STYLE_CATALOG):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        styled_u8 = apply_style(thumb_src, style=meta["id"], factor=3, upscale=3)
        tile_img = Image.fromarray(styled_u8).resize(
            (tile_width, tile_height), resample=Image.Resampling.NEAREST
        )
        canvas.paste(tile_img, (x0, y0 + header_h))
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + header_h - 1], fill=(14, 17, 36))
        label = f"{idx + 1:02d}. {meta['id'].upper()}"
        draw.text((x0 + 8, y0 + 7), label, fill=(0, 245, 212))

    canvas.save(output_path)
    return output_path
