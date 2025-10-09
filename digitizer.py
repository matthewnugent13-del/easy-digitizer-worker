# digitizer.py — build DST + the same stitch preview you see in Streamlit
# Reuses your modules from the files you uploaded.

from __future__ import annotations
from typing import Tuple, List, Dict
import io, os, math
from PIL import Image, ImageDraw

# pipeline pieces (your modules)
from preprocess import fit_to_hoop
from quantize import quantize_image
from vectorize import extract_color_regions, draw_contour_overlay
from ordering import compute_color_areas, clip_by_stack
from stitcher import (
    outline_then_fill_layers_universal,
    normalize_layers,
    plan_commands_from_layers,
    FILL_INSET_MM,
    MAX_RUN_TRAVEL_MM,
    MAX_JUMP_MM,
    RUN_STEP_MM,
)
from writer import save_dst

# pyembroidery for preview-from-stitches
from pyembroidery import read as read_emb, STITCH, JUMP, TRIM, COLOR_CHANGE, END

PX_PER_MM = 10.0  # must match your writer.py UNIT_SCALE assumptions (10 units/mm)

def _parse_pattern_by_color(pat) -> Dict:
    """(Copied/trimmed from your Streamlit app) turn EmbPattern -> polylines per step."""
    stitches = list(getattr(pat, "stitches", [])) or list(getattr(pat, "get_stitches", lambda: [])())
    steps, jumps, trims = [], [], []
    cur_step = {"polylines": [], "stitch_count": 0}
    cur_poly, cur_jump = [], []
    minx = miny = 1e18; maxx = maxy = -1e18

    def _flush_poly():
        nonlocal cur_poly, cur_step
        if len(cur_poly) >= 2:
            cur_step["polylines"].append(cur_poly)
        cur_poly = []

    def _flush_jump():
        nonlocal cur_jump, jumps
        if len(cur_jump) >= 2:
            jumps.append(cur_jump)
        cur_jump = []

    for (x, y, cmd) in stitches:
        x, y = float(x), float(y)
        minx = min(minx, x); maxx = max(maxx, x)
        miny = min(miny, y); maxy = max(maxy, y)

        if cmd == STITCH:
            cur_poly.append((x, y)); cur_step["stitch_count"] += 1
        elif cmd == JUMP:
            _flush_poly(); cur_jump.append((x, y))
        elif cmd == TRIM:
            _flush_poly(); _flush_jump(); trims.append((x, y))
        elif cmd == COLOR_CHANGE:
            _flush_poly(); _flush_jump()
            steps.append(cur_step if (cur_step["polylines"] or cur_step["stitch_count"]) else {"polylines": [], "stitch_count": 0})
            cur_step = {"polylines": [], "stitch_count": 0}
        elif cmd == END:
            break
        else:
            _flush_poly(); _flush_jump()

    _flush_poly(); _flush_jump()
    if cur_step["polylines"] or cur_step["stitch_count"] > 0:
        steps.append(cur_step)

    if minx > maxx or miny > maxy:
        minx = miny = 0.0; maxx = maxy = 1.0

    return {"steps": steps, "jumps": jumps, "trims": trims, "bounds": (minx, miny, maxx, maxy)}

def _render_stitch_preview(pat, width_px: int = 900, stroke_px: int = 2, flip_vertical: bool = False) -> Image.Image:
    """Draw stitches like your Streamlit DST viewer."""
    data = _parse_pattern_by_color(pat)
    steps, jumps, trims = data["steps"], data["jumps"], data["trims"]
    (minx, miny, maxx, maxy) = data["bounds"]
    w_units = max(1e-6, maxx - minx); h_units = max(1e-6, maxy - miny)
    aspect = h_units / w_units
    pad = 16; W = int(width_px); H = max(int(W * aspect), 200)
    canvas = Image.new("RGB", (W + pad*2, H + pad*2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    scale = W / w_units
    def tx(x: float) -> int: return int(pad + (x - minx) * scale)
    def ty(y: float) -> int:
        return int(pad + (maxy - y) * scale) if flip_vertical else int(pad + (y - miny) * scale)

    # vivid auto palette for visibility
    def _auto_colors(n: int) -> List[tuple[int,int,int]]:
        cols = []
        for i in range(max(1, n)):
            h = i / max(1, n)
            # simple HSV to RGB-ish
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            cols.append((int(r*255), int(g*255), int(b*255)))
        return cols

    colors = _auto_colors(len(steps))
    for i, step in enumerate(steps):
        col = colors[i % len(colors)]
        for pl in step["polylines"]:
            if len(pl) >= 2:
                pts = [(tx(x), ty(y)) for (x, y) in pl]
                draw.line(pts, fill=col, width=stroke_px, joint="curve")

    # jumps (light gray)
    for jpl in jumps:
        if len(jpl) >= 2:
            pts = [(tx(x), ty(y)) for (x, y) in jpl]
            draw.line(pts, fill=(180, 180, 180), width=max(1, stroke_px-1))

    # trims (red dots)
    for (x, y) in trims:
        cx, cy = tx(x), ty(y); r = max(3, stroke_px + 1)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(220, 0, 0), width=2)

    return canvas

def make_dst_and_preview(image_bytes: bytes) -> Tuple[bytes, bytes]:
    """Returns (dst_bytes, preview_png_bytes) using your exact pipeline + stitch viewer."""
    # 1) Load image & fit to hoop
    src = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    fitted = fit_to_hoop(src, hoop_mm=(130.0, 180.0), px_per_mm=PX_PER_MM)

    # 2) Quantize colors (keep whites, drop only transparent pixels)
    indexed_img, palette_rgb, _ = quantize_image(
        fitted, n_colors=6, remove_bg=True, ignore_alpha_only=True, alpha_threshold=8,
    )

    # 3) Vectorize regions + stack order + optional clipping
    regions = extract_color_regions(indexed_img, min_region_px=180, smooth_px=2.0)
    areas = compute_color_areas(regions)
    stack_order = sorted(areas.keys(), key=lambda k: areas[k], reverse=True)
    regions = clip_by_stack(regions, stack_order, margin_mm=0.05)

    # 4) Stitch layers (same defaults you used)
    layers_px = outline_then_fill_layers_universal(
        regions,
        px_per_mm=PX_PER_MM,
        outline_step_px=2.5,
        base_angle_deg=0.0,
        tatami_spacing_mm=0.50,
        narrow_fill_spacing_mm=0.60,
        satin_spacing_mm=0.40,
        angle_mode="per_color_random",
        angle_jitter_deg=0.0,
        serpentine=True,
        satin_max_width_mm=4.5,
        split_satin_max_width_mm=10.0,
        pull_comp_mm=0.30,
        tatami_bleed_mm=0.08,
    )
    layers_px = normalize_layers(layers_px, translate_origin="min")

    cmd_layers = plan_commands_from_layers(
        layers_px,
        px_per_mm=PX_PER_MM,
        max_run_mm=float(MAX_RUN_TRAVEL_MM),
        max_jump_mm=float(MAX_JUMP_MM),
        run_step_mm=float(RUN_STEP_MM),
        regions=regions,
        run_inside_regions_only=True,
        run_inset_mm=0.30,
    )

    # 5) Write DST to bytes (writer.save_dst needs a path, so use BytesIO temp file)
    tmp_path = "design.dst"
    save_dst(cmd_layers, tmp_path, palette_rgb=palette_rgb, color_order=stack_order, px_per_mm=PX_PER_MM)
    with open(tmp_path, "rb") as f:
        dst_bytes = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    # 6) Load pattern from bytes and render a PNG preview identical to your Streamlit viewer
    # pyembroidery requires a path, so write bytes to BytesIO -> temp path if needed:
    # Easiest: reuse tmp_path
    with open(tmp_path, "wb") as f:
        f.write(dst_bytes)
    pat = read_emb(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    img = _render_stitch_preview(pat, width_px=900, stroke_px=2, flip_vertical=False)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preview_png_bytes = buf.getvalue()

    return dst_bytes, preview_png_bytes
