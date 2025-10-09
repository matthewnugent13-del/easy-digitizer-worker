# app/ordering.py
from typing import Dict, List
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

PX_PER_MM = 10.0  # must match preprocess/stitcher

def compute_color_areas(regions: Dict[int, dict]) -> Dict[int, float]:
    areas = {}
    for k, data in regions.items():
        polys = data.get("polys", [])
        areas[k] = float(sum(p.area for p in polys))
    return areas

def _to_multipoly(polys: List[Polygon]) -> MultiPolygon:
    if not polys:
        return MultiPolygon([])
    u = unary_union(polys)
    if isinstance(u, Polygon):
        return MultiPolygon([u])
    return u

def clip_by_stack(
    regions: Dict[int, dict],
    order: List[int],
    margin_mm: float = 0.0,   # erosion applied to subtractor to avoid shaving too much
) -> Dict[int, dict]:
    """
    From TOP->BOTTOM 'order': each lower color is clipped by union of all above colors.
    margin_mm > 0 slightly erodes the subtractor so we leave a tiny moat instead of over-clipping.
    """
    margin_px = margin_mm * PX_PER_MM
    out = {k: {"polys": [p for p in v.get("polys", [])]} for k, v in regions.items()}

    seen_union = None  # union of all above layers (already processed)
    for color in order:  # top -> bottom
        polys = out[color].get("polys", [])
        if not polys:
            continue

        this_union = _to_multipoly(polys)

        if seen_union is not None and not seen_union.is_empty:
            sub = seen_union
            if margin_px > 0:
                sub = seen_union.buffer(-margin_px)
                if sub.is_empty:
                    sub = seen_union

            new_polys: List[Polygon] = []
            for p in polys:
                d = p.difference(sub)
                if d.is_empty:
                    continue
                if d.geom_type == "Polygon":
                    new_polys.append(d)
                else:
                    new_polys.extend([g for g in d.geoms if g.area > 1.0])
            out[color]["polys"] = new_polys

        if seen_union is None:
            seen_union = this_union
        else:
            seen_union = unary_union([seen_union, this_union])

    return out
