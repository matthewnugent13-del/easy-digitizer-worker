from typing import Dict, List, Tuple, Optional, Union, Iterable

# Support pyembroidery variants
try:
    from pyembroidery import (
        EmbPattern,
        STITCH,
        JUMP,
        TRIM,
        COLOR_CHANGE,
        END,
        EmbThread,
        write_dst,
    )
    HAVE_EMBTHREAD = True
except ImportError:
    from pyembroidery import (
        EmbPattern,
        STITCH,
        JUMP,
        TRIM,
        COLOR_CHANGE,
        END,
        write_dst,
    )
    HAVE_EMBTHREAD = False
    EmbThread = None  # type: ignore

# Write in 0.1 mm units so size matches hoop
UNIT_SCALE = 10.0   # 1 mm -> 10 DST units

Point      = Tuple[float, float]
Polyline   = List[Point]
CmdTuple   = Tuple[str, float, float]
LayerValue = Union[CmdTuple, Polyline]
LayerMap   = Dict[int, List[LayerValue]]

def _looks_like_cmd_tuple(x) -> bool:
    return (
        isinstance(x, tuple)
        and len(x) == 3
        and isinstance(x[0], str)
        and isinstance(x[1], (int, float))
        and isinstance(x[2], (int, float))
    )

def _looks_like_polyline(x) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        return False
    p0 = x[0]
    return isinstance(p0, (list, tuple)) and len(p0) == 2

def _is_polyline_layers(layers: LayerMap) -> bool:
    for vals in layers.values():
        for item in vals:
            if _looks_like_cmd_tuple(item):
                return False
            if _looks_like_polyline(item):
                return True
    return False

def _normalize_cmd_string(s: str) -> str:
    s = s.strip().lower()
    if not s:
        return "stitch"
    if s[0] == "j":
        return "jump"
    if s[0] == "t":
        return "trim"
    return "stitch"

def _polyline_layers_to_cmds_mm(layers_px: Dict[int, List[Polyline]], px_per_mm: float, first_point_cmd: str = "jump", next_point_cmd: str = "stitch") -> Dict[int, List[CmdTuple]]:
    if px_per_mm is None or px_per_mm <= 0:
        px_per_mm = 10.0
    out: Dict[int, List[CmdTuple]] = {}
    for color_idx, polylines in layers_px.items():
        cmds: List[CmdTuple] = []
        for pl in polylines:
            if not pl:
                continue
            x0, y0 = pl[0]
            cmds.append((_normalize_cmd_string(first_point_cmd), float(x0) / px_per_mm, float(y0) / px_per_mm))
            for (x, y) in pl[1:]:
                cmds.append((_normalize_cmd_string(next_point_cmd), float(x) / px_per_mm, float(y) / px_per_mm))
        out[color_idx] = cmds
    return out

def _to_cmd_layers_mm(layers: LayerMap, px_per_mm: float) -> Dict[int, List[CmdTuple]]:
    if _is_polyline_layers(layers):
        poly_layers: Dict[int, List[Polyline]] = {}
        for k, vals in layers.items():
            pls: List[Polyline] = []
            for v in vals:
                if _looks_like_polyline(v):
                    pls.append(list(v))
            poly_layers[k] = pls
        return _polyline_layers_to_cmds_mm(poly_layers, px_per_mm=px_per_mm)

    cmd_layers: Dict[int, List[CmdTuple]] = {}
    for k, vals in layers.items():
        conv: List[CmdTuple] = []
        for v in vals:
            if _looks_like_cmd_tuple(v):
                c, x, y = v
                conv.append((_normalize_cmd_string(c), float(x), float(y)))
        cmd_layers[k] = conv
    return cmd_layers

def save_dst(layers: LayerMap, path: str, palette_rgb: Optional[list] = None, color_order: Optional[List[int]] = None, *, px_per_mm: float = 10.0,) -> str:
    cmd_layers = _to_cmd_layers_mm(layers, px_per_mm=px_per_mm)

    pattern = EmbPattern()

    if color_order:
        order = [k for k in color_order if k in cmd_layers]
        for k in cmd_layers.keys():
            if k not in order:
                order.append(k)
    else:
        order = sorted(cmd_layers.keys())

    for i, color_idx in enumerate(order):
        if HAVE_EMBTHREAD:
            t = EmbThread()
            if palette_rgb and color_idx < len(palette_rgb):
                r, g, b = palette_rgb[color_idx]
                try:
                    t.set_color(r, g, b)
                except Exception:
                    pass
            pattern.add_thread(t)

        for cmd, x_mm, y_mm in cmd_layers[color_idx]:
            x = x_mm * UNIT_SCALE
            y = y_mm * UNIT_SCALE
            cmd_norm = _normalize_cmd_string(cmd)

            if cmd_norm == "jump":
                pattern.add_stitch_absolute(JUMP, x, y)
            elif cmd_norm == "trim":
                pattern.add_stitch_absolute(TRIM, x, y)
            else:
                pattern.add_stitch_absolute(STITCH, x, y)

        if i != len(order) - 1:
            pattern.add_command(COLOR_CHANGE)

    pattern.add_command(END)
    write_dst(pattern, path)
    return path
