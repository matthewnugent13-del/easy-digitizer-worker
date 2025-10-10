# digitizer.py — Easy Digitizer worker pipeline
# Generates DST bytes + a stitch preview PNG (like your Streamlit viewer).

from __future__ import annotations
from typing import Tuple, List
import io, os, math
from PIL import Image, ImageDraw

# ---- your modules (already in the repo root) ----
from preprocess import fit_to_hoop
from quantize import quantize_image
from vectorize import extract_color_regions
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

# Render stitch preview from DST
from pyembroidery import read as read_emb, STITCH, JUMP, TRIM, COLOR_CHANGE, END

# --------- speed / density controls ---------
PX_PER_MM = 7.0  # was 10.0; fewer points → faster

# --------- helper: parse EmbPattern into color steps / jumps / trims ---------
def _parse_pattern_by_color(pat) -> dict:
    stitches = list(getattr(pat, "stitches", []))
    if not stitches:
        try:
            stitches = list(pat.get_stitches())  # type: ignore
        except Exception:
            stitches = []

    steps: List[dict] = []
    jumps: List[List[tuple[float, float]]] = []
    trims: List[tuple[float, float]] = []

    cur_step = {"polylines": [], "stitch_count": 0}
    cur_poly: List[tuple[float, float]] = []
    cur_jump: List[tuple[float, float]] = []

    minx = miny = 1e18
    maxx = maxy = -1e18

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
        x = float(x); y = float(y)
        minx = min(minx, x); maxx = max(maxx, x)
        miny = min(miny, y); maxy = max(maxy, y)

        if cmd == STITCH:
            cur_poly.append((x, y))
            cur_step["stitch_count"] += 1
        elif cmd == JUMP:
            _flush_poly()
            cur_jump.append((x, y))
        elif cmd == TRIM:
            _flush_poly(); _flush_jump()
            trims.append((x, y))
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

# --------- helper: vivid stitch preview like Streamlit viewer ---------
def _render_stitch_preview(pat, width_px: int = 700, stroke_px: int = 2, flip_vertical: bool = False) -> Image.Image:
    data = _parse_pattern_by_color(pat)
    steps, jumps, trims = data["steps"], data["jumps"], data["trims"]
    (minx, miny, maxx, maxy) = data["bounds"]

    w_units = max(1e-6, maxx - minx)
    h_units = max(1e-6, maxy - miny)
    aspect = h_units / w_units
    pad = 16
    W = int(width_px)
    H = max(int(W * aspect), 200)

    canvas = Image.new("RGB", (W + pad*2, H + pad*2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    scale = W / w_units
    def tx(x: float) -> int: return int(pad + (x - minx) * scale)
    def ty(y: float) -> int:
        return int(pad + (maxy - y) * scale) if flip_vertical else int(pad + (y - miny) * scale)

    # distinct colors per step (simple HSV wheel)
    def _auto_colors(n: int) -> List[tuple[int,int,int]]:
        out: List[tuple[int,int,int]] = []
        for i in range(max(1, n)):
            h = i / max(1, n)
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            out.append((int(r*255), int(g*255), int(b*255)))
        return out

    cols = _auto_colors(len(steps))
    for i, step in enumerate(steps):
        col = cols[i % len(cols)]
        for pl in step["polylines"]:
            if len(pl) >= 2:
                draw.line([(tx(x), ty(y)) for (x, y) in pl], fill=col, width=stroke_px, joint="curve")

    # light gray for jumps
    for jpl in jumps:
        if len(jpl) >= 2:
            draw.line([(tx(x), ty(y)) for (x, y) in jpl], fill=(180, 180, 180), width=max(1, stroke_px-1))

    # red dots for trims
    for (x, y) in trims:
        cx, cy = tx(x), ty(y)
        r = max(3, stroke_px + 1)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(220, 0, 0), width=2)

    return canvas

# --------- main API called from main.py ---------
def make_dst_and_preview(image_bytes: bytes, n_colors: int = 6) -> Tuple[bytes, bytes]:
    """
    Returns (dst_bytes, preview_png_bytes)
    """
    # 1) Load & fit to hoop
    src = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    fitted = fit_to_hoop(src, hoop_mm=(130.0, 180.0), px_per_mm=PX_PER_MM)

    # 2) Quantize to 1..8 colors (fast path; falls back to Pillow if sklearn absent)
    n = max(1, min(8, int(n_colors)))
    indexed_img, palette_rgb, _ = quantize_image(
        fitted, n_colors=n, remove_bg=True, ignore_alpha_only=True, alpha_threshold=8
    )

    # 3) Vectorize (skip tiny regions for speed)
    regions = extract_color_regions(indexed_img, min_region_px=240, smooth_px=2.0)

    # 4) Stack & clip
    areas = compute_color_areas(regions)
    stack_order = sorted(areas.keys(), key=lambda k: areas[k], reverse=True)
    regions = clip_by_stack(regions, stack_order, margin_mm=0.05)

    # 5) Generate stitch layers
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

    cmds = plan_commands_from_layers(
        layers_px,
        px_per_mm=PX_PER_MM,
        max_run_mm=float(MAX_RUN_TRAVEL_MM),
        max_jump_mm=float(MAX_JUMP_MM),
        run_step_mm=float(RUN_STEP_MM),
        regions=regions,
        run_inside_regions_only=True,
        run_inset_mm=0.30,
    )

    # 6) Write DST to bytes
    tmp_path = "design.dst"
    save_dst(cmds, tmp_path, palette_rgb=palette_rgb, color_order=stack_order, px_per_mm=PX_PER_MM)
    with open(tmp_path, "rb") as f:
        dst_bytes = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    # 7) Render stitch preview from DST
    with open(tmp_path, "wb") as f:
        f.write(dst_bytes)
    pat = read_emb(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    img = _render_stitch_preview(pat, width_px=700, stroke_px=2, flip_vertical=False)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preview_png_bytes = buf.getvalue()

    return dst_bytes, preview_png_bytes
