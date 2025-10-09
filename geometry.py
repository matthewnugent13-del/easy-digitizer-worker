import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, LineString, box
from shapely import affinity

# --- Units: pixels in, pixels out. We'll convert to mm in the writer/stitcher. ---

def hatch_polygon_px(poly: Polygon, spacing_px: float = 5.0, angle_deg: float = 0.0) -> List[List[Tuple[float, float]]]:
    """
    Create parallel hatch lines clipped to 'poly'.
    Returns a list of polylines, each a list of (x_px, y_px).
    """
    if poly.is_empty or poly.area <= 0:
        return []

    # Rotate polygon so we can build horizontal lines
    rpoly = affinity.rotate(poly, angle=-angle_deg, origin='center')

    minx, miny, maxx, maxy = rpoly.bounds
    lines = []
    y = miny - (miny % spacing_px)

    # Build a stack of horizontal lines across the bbox
    while y <= maxy + spacing_px:
        ln = LineString([(minx - 10_000, y), (maxx + 10_000, y)])
        inter = ln.intersection(rpoly)
        if inter.is_empty:
            y += spacing_px
            continue

        if inter.geom_type == "LineString":
            coords = list(inter.coords)
            lines.append(coords)
        elif inter.geom_type == "MultiLineString":
            for seg in inter.geoms:
                coords = list(seg.coords)
                if len(coords) >= 2:
                    lines.append(coords)
        y += spacing_px

    # Rotate lines back
    out = []
    for coords in lines:
        ls = LineString(coords)
        ls = affinity.rotate(ls, angle=angle_deg, origin=rpoly.centroid)
        out.append([(float(x), float(y)) for (x, y) in ls.coords])

    return out
