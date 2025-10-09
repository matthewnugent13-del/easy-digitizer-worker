from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

from quantize import BACKGROUND_INDEX

def _mask_from_indexed(indexed: np.ndarray, color_idx: int) -> np.ndarray:
    return (indexed == color_idx).astype(np.uint8) * 255

def _polys_from_mask(mask: np.ndarray, smooth_px: float = 0.0, min_area_px: float = 100.0) -> List[Polygon]:
    if smooth_px and smooth_px > 0:
        k = max(1, int(round(smooth_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(contours) == 0:
        return []
    hier = hier[0]  # (N,4): next, prev, child, parent

    polys: List[Polygon] = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:  # skip holes here; add them as interiors
            continue
        ext = cnt[:, 0, :].astype(float)
        if len(ext) < 3: continue

        holes: List[List[Tuple[float, float]]] = []
        child = hier[i][2]
        while child != -1:
            hcnt = contours[child][:, 0, :].astype(float)
            if len(hcnt) >= 3:
                holes.append([(float(x), float(y)) for (x, y) in hcnt])
            child = hier[child][0]

        poly = Polygon([(float(x), float(y)) for (x, y) in ext], holes)
        if not poly.is_valid: poly = poly.buffer(0)
        if poly.is_empty or poly.area < float(min_area_px): continue
        polys.append(poly)

    if not polys: return []
    merged = unary_union(polys)
    if merged.is_empty: return []
    if merged.geom_type == "Polygon": return [merged]
    return [g for g in merged.geoms if g.area >= float(min_area_px)]

def extract_color_regions(indexed_img: Image.Image, min_region_px: int = 400, smooth_px: float = 2.0) -> Dict[int, dict]:
    if isinstance(indexed_img, Image.Image):
        indexed = np.array(indexed_img.convert("P")) if indexed_img.mode != "P" else np.array(indexed_img)
    else:
        indexed = np.asarray(indexed_img)

    colors = [int(c) for c in np.unique(indexed) if int(c) != BACKGROUND_INDEX]  # ← skip background
    regions: Dict[int, dict] = {}

    for idx in colors:
        mask = _mask_from_indexed(indexed, idx)
        polys = _polys_from_mask(mask, smooth_px=smooth_px, min_area_px=float(min_region_px))
        if polys:
            regions[idx] = {"polys": polys}
    return regions

def draw_contour_overlay(base_rgb_img, regions: Dict[int, dict], palette_rgb: List[Tuple[int, int, int]], line_w: int = 2):
    base = Image.fromarray(base_rgb_img.astype("uint8"), "RGB") if isinstance(base_rgb_img, np.ndarray) else base_rgb_img.convert("RGB")
    draw = ImageDraw.Draw(base)

    def _draw_poly(poly: Polygon, color):
        ex = list(poly.exterior.coords)
        if len(ex) > 1: draw.line(ex, fill=color, width=line_w, joint="curve")
        for ring in poly.interiors:
            pts = list(ring.coords)
            if len(pts) > 1: draw.line(pts, fill=color, width=line_w, joint="curve")

    for idx, data in regions.items():
        color = tuple(int(c) for c in palette_rgb[idx % len(palette_rgb)]) if palette_rgb else (255, 0, 0)
        if "polys" in data and data["polys"]:
            for poly in data["polys"]: _draw_poly(poly, color)
    return base

def remove_canvas_background(regions: Dict[int, dict], canvas_w: int, canvas_h: int, margin_px: int = 3, coverage_ratio: float = 0.50):
    total_area = float(canvas_w * canvas_h)
    dropped, out = [], {}
    for idx, data in regions.items():
        polys = data.get("polys", [])
        if not polys: continue
        area = sum(p.area for p in polys)
        touches = any(
            (p.bounds[0] <= margin_px) or (p.bounds[1] <= margin_px) or
            (p.bounds[2] >= canvas_w - margin_px) or (p.bounds[3] >= canvas_h - margin_px)
            for p in polys
        )
        if touches and (area / total_area >= coverage_ratio):
            dropped.append(idx)
        else:
            out[idx] = {"polys": polys[:]}

    return out, dropped
