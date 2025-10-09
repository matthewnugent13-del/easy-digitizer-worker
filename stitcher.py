# stitcher.py
# Universal fill engine with: outlines → satin/narrow/tatami selection,
# serpentine hatching, underlays, pull compensation, per-color angles,
# normalization, and safe travel planning.

from typing import Dict, List, Tuple, Optional, Union, Any
import math
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.affinity import rotate
from shapely.ops import unary_union
from shapely import wkt

# -----------------------------
# Public constants (UI may import these)
# -----------------------------
FILL_INSET_MM: float = 0.10
DEFAULT_BASE_ANGLE_DEG: float = 0.0
PX_PER_MM_DEFAULT: float = 10.0

# Travel defaults
MAX_RUN_TRAVEL_MM: float = 6.0    # run (hidden) if <= this AND inside same-color region
RUN_STEP_MM: float = 2.8
MAX_JUMP_MM: float = 14.0         # jump if <= this; otherwise trim + jump

# Satin / fill thresholds (presets may override)
SATIN_MAX_WIDTH_MM: float = 4.0
SPLIT_SATIN_MAX_WIDTH_MM: float = 10.0

# -----------------------------
# Types
# -----------------------------
Point = Tuple[float, float]
Polyline = List[Point]
RegionMap = Dict[int, Any]            # color_idx -> list[Polygon-like] or legacy dicts
LayerMap = Dict[int, List[Polyline]]  # color_idx -> list of polylines
CmdTuple = Tuple[str, float, float]   # ("stitch"|"jump"|"trim", x_mm, y_mm)
CmdLayers = Dict[int, List[CmdTuple]]

# ============================================================
# Helpers
# ============================================================
def _stable_rand01(n: int) -> float:
    """Deterministic pseudo-random in [0,1) from an int."""
    n = (n ^ 0x9E3779B9) & 0xFFFFFFFF
    n = (n * 2654435761) & 0xFFFFFFFF
    n ^= (n >> 16)
    return ((n & 0xFFFFFFFF) / 2**32)

def _per_color_angle_map(color_keys, base_angle_deg: float = 0.0) -> dict[int, float]:
    """Assign one angle per color; different colors get different angles."""
    ks = [int(k) for k in color_keys]
    if not ks:
        return {}
    seed = sum(int(k) * 131 for k in ks)     # stable palette seed
    offset = _stable_rand01(seed) * 180.0    # deterministic “random” offset
    step = 180.0 / len(ks)                   # spread evenly
    return {int(k): (base_angle_deg + offset + i * step) % 180.0 for i, k in enumerate(ks)}

def _coerce_to_polygon(obj: Any) -> Optional[Polygon]:
    if obj is None:
        return None
    if isinstance(obj, Polygon):
        return obj
    if hasattr(obj, "geom_type") and getattr(obj, "geom_type", "") == "Polygon":
        return Polygon(obj.exterior.coords) if not isinstance(obj, Polygon) else obj
    if isinstance(obj, str):
        try:
            geom = wkt.loads(obj)
            if isinstance(geom, Polygon):
                return geom
            if hasattr(geom, "convex_hull"):
                hull = geom.convex_hull
                if isinstance(hull, Polygon):
                    return hull
        except Exception:
            return None
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        try:
            first = obj[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                return Polygon(obj)
        except Exception:
            return None
    return None

def _extract_poly_list(entry: Any) -> List[Any]:
    if isinstance(entry, dict):
        if "polys" in entry and isinstance(entry["polys"], (list, tuple)):
            return list(entry["polys"])
        if "polygons" in entry and isinstance(entry["polygons"], (list, tuple)):
            return list(entry["polygons"])
        if "geometry" in entry:
            g = entry["geometry"]
            return [g] if not isinstance(g, (list, tuple)) else list(g)
        return []
    if isinstance(entry, (list, tuple)):
        return list(entry)
    return [entry]

# ============================================================
# Outlines
# ============================================================
def _ring_to_running_stitch(coords: List[Tuple[float, float]], step_px: float) -> Polyline:
    if len(coords) < 2:
        return []
    if coords[0] != coords[-1]:
        coords = list(coords) + [coords[0]]
    dists = [0.0]
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        dists.append(dists[-1] + math.hypot(x1 - x0, y1 - y0))
    total = dists[-1]
    if total <= 0:
        return [coords[0]]
    num_pts = max(2, int(total / max(1e-6, step_px)) + 1)
    targets = [i * total / (num_pts - 1) for i in range(num_pts)]
    out: Polyline = []
    seg_idx = 0
    for t in targets:
        while seg_idx + 1 < len(dists) and dists[seg_idx + 1] < t:
            seg_idx += 1
        if seg_idx + 1 >= len(dists):
            out.append(coords[-1]); continue
        t0, t1 = dists[seg_idx], dists[seg_idx + 1]
        x0, y0 = coords[seg_idx]; x1, y1 = coords[seg_idx + 1]
        if t1 - t0 <= 1e-9:
            out.append((x0, y0))
        else:
            a = (t - t0) / (t1 - t0)
            out.append((x0 + a * (x1 - x0), y0 + a * (y1 - y0)))
    return out

def _polygon_outline_running_stitches(poly: Polygon, step_px: float) -> List[Polyline]:
    res: List[Polyline] = []
    if poly.is_empty:
        return res
    try:
        for interior in poly.interiors:
            res.append(_ring_to_running_stitch(list(interior.coords), step_px))
        res.append(_ring_to_running_stitch(list(poly.exterior.coords), step_px))
    except Exception:
        hull = poly.convex_hull
        res.append(_ring_to_running_stitch(list(hull.exterior.coords), step_px))
    return [pline for pline in res if len(pline) >= 2]

# ============================================================
# Hatching (rotate polygon −angle, clip, rotate rows back +angle)
# ============================================================
def _densify_line(line: LineString, step_px: float) -> Polyline:
    if line.is_empty:
        return []
    length = float(line.length)
    if length <= 0:
        x, y = line.coords[0]
        return [(x, y)]
    n = max(2, int(length / max(1e-6, step_px)) + 1)
    distances = np.linspace(0.0, length, n)
    pts = [line.interpolate(float(d)).coords[0] for d in distances]
    return [(float(x), float(y)) for (x, y) in pts]

def _dominant_angle_deg(poly: Polygon) -> float:
    xs, ys = poly.exterior.xy
    pts = np.column_stack([xs, ys]).astype(float)
    if len(pts) < 3:
        return 0.0
    mu = pts.mean(axis=0)
    cov = np.cov((pts - mu).T)
    evals, evecs = np.linalg.eig(cov)
    vec = evecs[:, int(np.argmax(evals))]
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    return angle % 180.0

def _stable_jitter_deg(poly: Polygon, jitter_range_deg: float) -> float:
    if jitter_range_deg <= 0:
        return 0.0
    cx, cy = poly.centroid.x, poly.centroid.y
    seed = math.sin(cx * 12.9898 + cy * 78.233) * 43758.5453
    frac = seed - math.floor(seed)
    return (frac * 2.0 - 1.0) * jitter_range_deg

def _hatch_segments(
    poly: Polygon,
    spacing_px: float,
    angle_deg: float,
    inset_px: float,
    bleed_px: float,
    step_px: float,
    serpentine: bool = True,
    pull_comp_px: float = 0.0,
) -> List[Polyline]:
    """Rotate polygon by -angle, keep lines horizontal, clip, rotate rows back +angle."""
    work = poly
    if pull_comp_px != 0.0:
        work = work.buffer(float(pull_comp_px))
        if work.is_empty:
            work = poly

    inset_poly = work.buffer(-inset_px)
    if inset_poly.is_empty:
        return []
    target_poly = inset_poly.buffer(bleed_px) if bleed_px else inset_poly

    minx, miny, maxx, maxy = target_poly.bounds
    center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    rpoly = rotate(target_poly, -angle_deg, origin=center, use_radians=False)

    rminx, rminy, rmaxx, rmaxy = rpoly.bounds
    diag = float(math.hypot(rmaxx - rminx, rmaxy - rminy)) + 2.0 * spacing_px
    num = int((2 * diag + (rmaxy - rminy) + (rmaxx - rminx)) / spacing_px) + 2

    lines = []
    y0 = rminy - diag
    for i in range(-num, num + 1):
        y = y0 + i * spacing_px
        lines.append(LineString([(rminx - diag, y), (rmaxx + diag, y)]))

    # clip lines to rotated polygon
    rows: List[Polyline] = []
    row_keys: List[Tuple[int, float]] = []
    for idx, ln in enumerate(lines):
        inter = rpoly.intersection(ln)
        if inter.is_empty:
            continue
        if isinstance(inter, LineString):
            pl = _densify_line(inter, step_px)
            if len(pl) >= 2:
                rows.append(pl); row_keys.append((len(rows)-1, float(ln.coords[0][1])))
        elif isinstance(inter, MultiLineString):
            for seg in inter.geoms:
                if not seg.is_empty:
                    pl = _densify_line(seg, step_px)
                    if len(pl) >= 2:
                        rows.append(pl); row_keys.append((len(rows)-1, float(ln.coords[0][1])))

    # serpentine order
    if rows and serpentine:
        order = sorted(row_keys, key=lambda t: t[1])
        serp: List[Polyline] = []
        flip = False
        for i, _ in order:
            r = rows[i]
            serp.append(list(reversed(r)) if flip else r)
            flip = not flip
        rows = serp

    # rotate rows back
    if abs(angle_deg) > 1e-9:
        cos_a = math.cos(math.radians(angle_deg))
        sin_a = math.sin(math.radians(angle_deg))
        cx, cy = center
        def rot_back(pt: Point) -> Point:
            x, y = pt
            dx, dy = (x - cx), (y - cy)
            return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        rows = [[rot_back(p) for p in r] for r in rows]

    return rows

# ============================================================
# Public API — Fills / Outlines (supports per_color_random)
# ============================================================
def fills_from_regions(
    regions: RegionMap,
    spacing_mm: float,
    base_angle_deg: float,
    edge_inset_mm: float,
    edge_bleed_mm: float,
    px_per_mm: Optional[float] = None,
    step_px: Optional[float] = None,
    *,
    angle_mode: str = "per_region",   # "global" | "per_region" | "alternate" | "per_color_random"
    angle_jitter_deg: float = 6.0,
    serpentine: bool = True,
    pull_comp_mm: float = 0.0,
) -> LayerMap:
    if not regions:
        return {}
    if px_per_mm is None:
        px_per_mm = PX_PER_MM_DEFAULT

    spacing_px   = float(spacing_mm) * px_per_mm
    inset_px     = float(edge_inset_mm) * px_per_mm
    bleed_px     = float(edge_bleed_mm) * px_per_mm
    pull_comp_px = float(pull_comp_mm) * px_per_mm
    if step_px is None:
        step_px = max(1.0, spacing_px * 0.5)

    angle_mode = (angle_mode or "global").strip().lower()
    per_color_random_angles = None
    if angle_mode == "per_color_random":
        per_color_random_angles = _per_color_angle_map(regions.keys(), base_angle_deg)

    result: LayerMap = {}
    for color_idx, entry in regions.items():
        polylines: List[Polyline] = []
        items = _extract_poly_list(entry)
        for idx, item in enumerate(items):
            poly = _coerce_to_polygon(item)
            if poly is None or poly.is_empty:
                continue

            if angle_mode == "per_region":
                angle = _dominant_angle_deg(poly)
            elif angle_mode == "alternate":
                angle = base_angle_deg + (90.0 if (idx % 2 == 1) else 0.0)
            elif angle_mode == "per_color_random":
                angle = per_color_random_angles.get(int(color_idx), base_angle_deg)
            else:  # "global"
                angle = base_angle_deg

            angle = (angle % 180.0) + _stable_jitter_deg(poly, angle_jitter_deg)

            polylines.extend(
                _hatch_segments(
                    poly, spacing_px, angle, inset_px, bleed_px, step_px,
                    serpentine=serpentine, pull_comp_px=pull_comp_px
                )
            )
        result[color_idx] = polylines
    return result

def contours_to_running_layers(regions: RegionMap, step_px: float) -> LayerMap:
    layers: LayerMap = {}
    for color_idx, entry in regions.items():
        polylines: List[Polyline] = []
        for item in _extract_poly_list(entry):
            poly = _coerce_to_polygon(item)
            if poly is None or poly.is_empty:
                continue
            # holes then exterior
            for interior in poly.interiors:
                polylines.append(_ring_to_running_stitch(list(interior.coords), step_px))
            polylines.append(_ring_to_running_stitch(list(poly.exterior.coords), step_px))
        layers[color_idx] = [pl for pl in polylines if len(pl) >= 2]
    return layers

# ============================================================
# Shape classification → satin / narrow fill / tatami
# ============================================================
def _min_dim_mm(poly: Polygon, px_per_mm: float) -> float:
    minx, miny, maxx, maxy = poly.bounds
    w_mm = (maxx - minx) / px_per_mm
    h_mm = (maxy - miny) / px_per_mm
    return float(min(w_mm, h_mm))

def outline_then_fill_layers_universal(
    regions: RegionMap,
    *,
    px_per_mm: float = PX_PER_MM_DEFAULT,
    outline_step_px: float = 2.5,
    base_angle_deg: float = 0.0,
    # fill controls
    tatami_spacing_mm: float = 0.50,
    tatami_inset_mm: float = FILL_INSET_MM,
    tatami_bleed_mm: float = 0.08,
    narrow_fill_spacing_mm: float = 0.60,
    satin_spacing_mm: float = 0.40,
    # angle controls
    angle_mode: str = "per_region",
    angle_jitter_deg: float = 6.0,
    serpentine: bool = True,
    # thresholds
    satin_max_width_mm: float = SATIN_MAX_WIDTH_MM,
    split_satin_max_width_mm: float = SPLIT_SATIN_MAX_WIDTH_MM,
    # compensation
    pull_comp_mm: float = 0.30,
) -> LayerMap:
    outlines = contours_to_running_layers(regions, step_px=float(outline_step_px))

    satin_regions: RegionMap = {}
    narrow_regions: RegionMap = {}
    tatami_regions: RegionMap = {}

    for color_idx, entry in regions.items():
        items = _extract_poly_list(entry)
        for item in items:
            poly = _coerce_to_polygon(item)
            if poly is None or poly.is_empty:
                continue
            wmin = _min_dim_mm(poly, px_per_mm=px_per_mm)
            if wmin < satin_max_width_mm:
                satin_regions.setdefault(color_idx, []).append(poly)
            elif wmin < split_satin_max_width_mm:
                narrow_regions.setdefault(color_idx, []).append(poly)
            else:
                tatami_regions.setdefault(color_idx, []).append(poly)

    layers: LayerMap = {}

    def _edge_walk_underlay(polys: List[Polygon], inset_mm: float, step_px: float):
        res: List[Polyline] = []
        inset_px = inset_mm * px_per_mm
        for p in polys:
            inner = p.buffer(-inset_px)
            if inner.is_empty:
                inner = p
            try:
                res.append(_ring_to_running_stitch(list(inner.exterior.coords), step_px))
            except Exception:
                pass
        return [r for r in res if len(r) >= 2]

    def _center_run_underlay(polys: List[Polygon], step_px: float):
        res: List[Polyline] = []
        for p in polys:
            try:
                res.append(_ring_to_running_stitch(list(p.exterior.coords), step_px * 1.4))
            except Exception:
                pass
        return [r for r in res if len(r) >= 2]

    # SATIN-like + center-run underlay
    if satin_regions:
        under = {k: _center_run_underlay(v, step_px=float(outline_step_px)) for k, v in satin_regions.items()}
        fills = fills_from_regions(
            satin_regions, spacing_mm=float(satin_spacing_mm), base_angle_deg=base_angle_deg,
            edge_inset_mm=float(FILL_INSET_MM) * 0.7, edge_bleed_mm=float(tatami_bleed_mm) * 0.5,
            px_per_mm=float(px_per_mm), step_px=float(outline_step_px) * 0.6,
            angle_mode=angle_mode, angle_jitter_deg=angle_jitter_deg, serpentine=serpentine,
            pull_comp_mm=float(pull_comp_mm) * 1.1,
        )
        for k in set(under.keys()) | set(fills.keys()):
            seq: List[Polyline] = []
            seq.extend(under.get(k, [])); seq.extend(fills.get(k, []))
            layers.setdefault(k, []).extend(seq)

    # NARROW fill + edge-walk underlay
    if narrow_regions:
        under = {k: _edge_walk_underlay(v, inset_mm=float(FILL_INSET_MM)*0.8, step_px=float(outline_step_px)) for k, v in narrow_regions.items()}
        fills = fills_from_regions(
            narrow_regions, spacing_mm=float(narrow_fill_spacing_mm), base_angle_deg=base_angle_deg,
            edge_inset_mm=float(FILL_INSET_MM), edge_bleed_mm=float(tatami_bleed_mm),
            px_per_mm=float(px_per_mm), step_px=float(outline_step_px) * 0.75,
            angle_mode=angle_mode, angle_jitter_deg=angle_jitter_deg, serpentine=serpentine,
            pull_comp_mm=float(pull_comp_mm),
        )
        for k in set(under.keys()) | set(fills.keys()):
            seq: List[Polyline] = []
            seq.extend(under.get(k, [])); seq.extend(fills.get(k, []))
            layers.setdefault(k, []).extend(seq)

    # TATAMI + edge-walk underlay
    if tatami_regions:
        under = {k: _edge_walk_underlay(v, inset_mm=float(FILL_INSET_MM), step_px=float(outline_step_px)) for k, v in tatami_regions.items()}
        fills = fills_from_regions(
            tatami_regions, spacing_mm=float(tatami_spacing_mm), base_angle_deg=base_angle_deg,
            edge_inset_mm=float(FILL_INSET_MM), edge_bleed_mm=float(tatami_bleed_mm),
            px_per_mm=float(px_per_mm), step_px=float(outline_step_px) * 0.8,
            angle_mode=angle_mode, angle_jitter_deg=angle_jitter_deg, serpentine=serpentine,
            pull_comp_mm=float(pull_comp_mm),
        )
        for k in set(under.keys()) | set(fills.keys()):
            seq: List[Polyline] = []
            seq.extend(under.get(k, [])); seq.extend(fills.get(k, []))
            layers.setdefault(k, []).extend(seq)

    # Finally: geometry outlines
    for k, seq in (outlines or {}).items():
        layers.setdefault(k, []).extend(seq)

    return layers

# ============================================================
# Normalization (before travel planning)
# ============================================================
def normalize_layers(
    layers: LayerMap,
    *args,
    translate_origin: Union[str, Tuple[float, float]] = 'min',
    scale: float = 1.0,
    round_to: Optional[float] = None,
    max_jump: Optional[float] = None,
    **kwargs
) -> LayerMap:
    xs, ys = [], []
    for polylines in layers.values():
        for pl in polylines:
            for (x, y) in pl:
                xs.append(x); ys.append(y)
    if not xs or not ys:
        return layers
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    if isinstance(translate_origin, str):
        if translate_origin == 'min':
            tx, ty = -minx, -miny
        elif translate_origin == 'center':
            tx, ty = -cx, -cy
        else:
            tx, ty = 0.0, 0.0
    elif isinstance(translate_origin, (tuple, list)) and len(translate_origin) == 2:
        tx, ty = -float(translate_origin[0]), -float(translate_origin[1])
    else:
        tx, ty = 0.0, 0.0

    def _round_val(v: float) -> float:
        if round_to and round_to > 0:
            return round(v / round_to) * round_to
        return v

    def _split_by_jump(polyline: Polyline, max_jump_dist: float) -> List[Polyline]:
        if max_jump_dist is None or len(polyline) < 2:
            return [polyline]
        out: List[Polyline] = []
        cur: Polyline = [polyline[0]]
        for i in range(1, len(polyline)):
            x0, y0 = cur[-1]; x1, y1 = polyline[i]
            if math.hypot(x1 - x0, y1 - y0) > max_jump_dist and len(cur) >= 2:
                out.append(cur); cur = [polyline[i]]
            else:
                cur.append(polyline[i])
        if len(cur) >= 2:
            out.append(cur)
        return out

    new_layers: LayerMap = {}
    for color_idx, polylines in layers.items():
        new_polys: List[Polyline] = []
        for pl in polylines:
            if not pl:
                continue
            tr = [(_round_val((x + tx) * scale), _round_val((y + ty) * scale)) for (x, y) in pl]
            if max_jump is not None and max_jump > 0:
                new_polys.extend(_split_by_jump(tr, max_jump))
            else:
                new_polys.append(tr)
        new_layers[color_idx] = new_polys
    return new_layers

# ============================================================
# Travel planning — SAFE hidden runs (only inside same-color coverage)
# ============================================================
def _dist(a: Point, b: Point) -> float:
    dx, dy = (a[0]-b[0], a[1]-b[1])
    return math.hypot(dx, dy)

def _nearest_chain(polylines: List[Polyline]) -> List[Polyline]:
    if not polylines:
        return []
    unused = [(i, pl) for i, pl in enumerate(polylines)]
    allpts = [pt for pl in polylines for pt in (pl[0], pl[-1])]
    cx = sum(p[0] for p in allpts)/len(allpts)
    cy = sum(p[1] for p in allpts)/len(allpts)
    start_idx = min(range(len(unused)),
                    key=lambda j: min(_dist((cx,cy), unused[j][1][0]),
                                      _dist((cx,cy), unused[j][1][-1])))
    cur = unused.pop(start_idx)[1]
    if _dist((cx,cy), cur[-1]) < _dist((cx,cy), cur[0]):
        cur = list(reversed(cur))
    path = [cur]
    cur_end = cur[-1]
    while unused:
        best_k = None
        best_rev = False
        best_d = 1e18
        for k,(i,pl) in enumerate(unused):
            d0 = _dist(cur_end, pl[0])
            d1 = _dist(cur_end, pl[-1])
            if d0 < best_d:
                best_d, best_k, best_rev = d0, k, False
            if d1 < best_d:
                best_d, best_k, best_rev = d1, k, True
        _, nxt = unused.pop(best_k)
        if best_rev:
            nxt = list(reversed(nxt))
        path.append(nxt)
        cur_end = nxt[-1]
    return path
def _chain_preferring_hidden_runs(
    polylines: List[Polyline],
    color_idx: int,
    cover_by_color,
    px_per_mm: float,
    max_run_mm: float,
) -> List[Polyline]:
    """Order polylines preferring next segments reachable by a safe hidden run."""
    if not polylines:
        return []
    # start near centroid (same spirit as _nearest_chain)
    allpts = [pt for pl in polylines for pt in (pl[0], pl[-1])]
    cx = sum(p[0] for p in allpts) / len(allpts)
    cy = sum(p[1] for p in allpts) / len(allpts)

    def _dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    unused = [(i, list(pl)) for i, pl in enumerate(polylines) if len(pl) >= 2]
    start_idx = min(
        range(len(unused)),
        key=lambda j: min(_dist((cx, cy), unused[j][1][0]),
                          _dist((cx, cy), unused[j][1][-1])),
    )
    cur = unused.pop(start_idx)[1]
    if _dist((cx, cy), cur[-1]) < _dist((cx, cy), cur[0]):
        cur = list(reversed(cur))
    path = [cur]
    cur_end = cur[-1]

    cov = cover_by_color.get(color_idx)

    def _can_hide(a_px, b_px) -> bool:
        if cov is None:
            return False
        try:
            # tolerant: boundary is OK
            return cov.covers(LineString([a_px, b_px]))
        except Exception:
            return False

    while unused:
        best = None
        best_rev = False
        best_score = 1e18
        for k, pl in unused:
            for endpoint, rev in ((pl[0], False), (pl[-1], True)):
                d_mm = _dist(cur_end, endpoint) / px_per_mm
                # priority: 0 (hidden run possible) < 1 (short jump) < 2 (long move)
                bucket = 0 if (d_mm <= max_run_mm and _can_hide(cur_end, endpoint)) else (1 if d_mm <= 14.0 else 2)
                score = bucket * 1_000_000 + d_mm
                if score < best_score:
                    best, best_rev, best_score = k, rev, score
        # take the best candidate
        idx = next(i for i, (k, _) in enumerate(unused) if k == best)
        nxt = unused.pop(idx)[1]
        if best_rev:
            nxt = list(reversed(nxt))
        path.append(nxt)
        cur_end = nxt[-1]
    return path

def _straight_run(a: Point, b: Point, step_px: float) -> Polyline:
    dist = _dist(a, b)
    if dist <= step_px:
        return [a, b]
    n = max(2, int(dist / step_px) + 1)
    out = []
    for i in range(n):
        t = i/(n-1)
        out.append((a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t))
    return out

def plan_commands_from_layers(
    layers_px: LayerMap,
    *,
    px_per_mm: float = PX_PER_MM_DEFAULT,
    max_run_mm: float = MAX_RUN_TRAVEL_MM,
    max_jump_mm: float = MAX_JUMP_MM,
    run_step_mm: float = RUN_STEP_MM,
    # Safe hidden-travel controls
    regions: Optional[RegionMap] = None,
    run_inside_regions_only: bool = True,
    run_inset_mm: float = 0.30,
    allow_trims: bool = True,   # NEW
) -> CmdLayers:
    """
    Convert polyline layers (pixels) into command tuples (mm) with travel planning.
    Policy: prefer hidden runs; then short jumps; trims only if allowed and move is long.
    """
    cmd_layers: CmdLayers = {}
    run_step_px = max(1.0, run_step_mm * px_per_mm)

    # Build a "safe coverage" shape per color (union of regions, slightly shrunken)
    cover_by_color: Dict[int, Any] = {}
    if run_inside_regions_only and regions:
        shrink_px = float(run_inset_mm) * px_per_mm
        for color_idx, entry in regions.items():
            polys = []
            for item in _extract_poly_list(entry):
                p = _coerce_to_polygon(item)
                if p is not None and not p.is_empty:
                    polys.append(p)
            if polys:
                try:
                    u = unary_union(polys)
                    u_shr = u.buffer(-shrink_px)
                    cover_by_color[color_idx] = (u_shr if not u_shr.is_empty else u)
                except Exception:
                    cover_by_color[color_idx] = unary_union(polys)

    def _emit_run_between(cmds: List[CmdTuple], a_px: Point, b_px: Point):
        pl = _straight_run(a_px, b_px, step_px=run_step_px)
        for (x, y) in pl:
            cmds.append(("stitch", float(x)/px_per_mm, float(y)/px_per_mm))

    def _can_hide_run(start_px: Point, end_px: Point, color_idx: int) -> bool:
        if not run_inside_regions_only:
            return True
        cov = cover_by_color.get(color_idx)
        if cov is None:
            return False
        try:
            # tolerant: boundary is OK
            return cov.covers(LineString([start_px, end_px]))
        except Exception:
            return False

    for color_idx, polylines in layers_px.items():
        seq = [pl for pl in polylines if len(pl) >= 2]
        if not seq:
            cmd_layers[color_idx] = []
            continue

        # Prefer runs when ordering
        ordered = _chain_preferring_hidden_runs(
            seq, color_idx=color_idx, cover_by_color=cover_by_color,
            px_per_mm=px_per_mm, max_run_mm=max_run_mm
        )

        cmds: List[CmdTuple] = []
        cur_pt: Optional[Point] = None

        for pl in ordered:
            start = pl[0]
            if cur_pt is not None:
                gap_mm = math.hypot(cur_pt[0]-start[0], cur_pt[1]-start[1]) / px_per_mm

                if gap_mm <= max_run_mm and _can_hide_run(cur_pt, start, color_idx):
                    _emit_run_between(cmds, cur_pt, start)
                elif gap_mm <= max_jump_mm:
                    cmds.append(("jump", float(start[0])/px_per_mm, float(start[1])/px_per_mm))
                else:
                    if allow_trims:
                        cmds.append(("trim", float(cur_pt[0])/px_per_mm, float(cur_pt[1])/px_per_mm))
                    cmds.append(("jump", float(start[0])/px_per_mm, float(start[1])/px_per_mm))
            else:
                # first move for this color
                cmds.append(("jump", float(start[0])/px_per_mm, float(start[1])/px_per_mm))

            # sew this polyline body
            for (x, y) in pl[1:]:
                cmds.append(("stitch", float(x)/px_per_mm, float(y)/px_per_mm))
            cur_pt = pl[-1]

        cmd_layers[color_idx] = cmds

    return cmd_layers


    def _can_hide_run(start_px: Point, end_px: Point, color_idx: int) -> bool:
        if not run_inside_regions_only:
            return True
        cov = cover_by_color.get(color_idx)
        if cov is None:
            return False
        try:
            path = LineString([start_px, end_px])
            return cov.contains(path)
        except Exception:
            return False

    def _emit_run_between(cmds: List[CmdTuple], a_px: Point, b_px: Point):
        run_pl = _straight_run(a_px, b_px, step_px=run_step_px)
        for (x, y) in run_pl:
            cmds.append(("stitch", float(x)/px_per_mm, float(y)/px_per_mm))

    for color_idx, polylines in layers_px.items():
        ordered = _chain_preferring_hidden_runs(
    [pl for pl in polylines if len(pl) >= 2],
    color_idx=color_idx,
    cover_by_color=cover_by_color,
    px_per_mm=px_per_mm,
    max_run_mm=max_run_mm,
)
        cmds: List[CmdTuple] = []
        cur_pt: Optional[Point] = None

        for pl in ordered:
            start = pl[0]
            if cur_pt is not None:
                gap_mm = _dist(cur_pt, start) / px_per_mm

                # SAFE: hidden run only inside same-color coverage and within run length
                if gap_mm <= max_run_mm and _can_hide_run(cur_pt, start, color_idx):
                    _emit_run_between(cmds, cur_pt, start)
                elif gap_mm <= max_jump_mm:
                    cmds.append(("jump", float(start[0])/px_per_mm, float(start[1])/px_per_mm))
                else:
                    cmds.append(("trim", float(cur_pt[0])/px_per_mm, float(cur_pt[1])/px_per_mm))
                    cmds.append(("jump", float(start[0])/px_per_mm, float(start[1])/px_per_mm))
            else:
                # first move for this color
                cmds.append(("jump", float(start[0])/px_per_mm, float(start[1])/px_per_mm))

            # sew this polyline
            for (x, y) in pl[1:]:
                cmds.append(("stitch", float(x)/px_per_mm, float(y)/px_per_mm))
            cur_pt = pl[-1]

        cmd_layers[color_idx] = cmds

    return cmd_layers
