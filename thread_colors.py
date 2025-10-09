# thread_colors.py
# Minimal thread palettes and mapping to the nearest brand color.
from __future__ import annotations
from typing import List, Dict, Tuple
import math

RGB = Tuple[int, int, int]

def _srgb_to_linear(x: float) -> float:
    x = x / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

def _rgb_to_xyz(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = rgb
    r = _srgb_to_linear(r); g = _srgb_to_linear(g); b = _srgb_to_linear(b)
    # sRGB D65
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z

def _xyz_to_lab(xyz: Tuple[float,float,float]) -> Tuple[float,float,float]:
    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz
    x /= Xn; y /= Yn; z /= Zn
    def f(t: float) -> float:
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16.0/116.0)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return (L, a, b)

def _rgb_to_lab(rgb: RGB) -> Tuple[float,float,float]:
    return _xyz_to_lab(_rgb_to_xyz(rgb))

def _de76(lab1, lab2) -> float:
    return math.sqrt((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2)

def _hex(rgb: RGB) -> str:
    return '#%02X%02X%02X' % rgb

# ---- Minimal, practical thread palettes ----
THREAD_BRANDS: Dict[str, Dict] = {
    "Generic 40wt (basic)": {
        "desc": "Generic, balanced 40wt set (~48 colors)",
        "colors": [
            # neutrals
            ("GEN-000", "Black",          (0, 0, 0)),
            ("GEN-001", "White",          (255, 255, 255)),
            ("GEN-002", "Very Light Gray",(225, 225, 225)),
            ("GEN-003", "Light Gray",     (200, 200, 200)),
            ("GEN-004", "Medium Gray",    (160, 160, 160)),
            ("GEN-005", "Dark Gray",      (96, 96, 96)),
            # yellows / golds
            ("GEN-100", "Lemon",          (255, 247, 79)),
            ("GEN-101", "Yellow",         (255, 224, 0)),
            ("GEN-102", "Golden Yellow",  (246, 190, 0)),
            ("GEN-103", "Gold",           (218, 165, 32)),
            ("GEN-104", "Mustard",        (204, 153, 0)),
            # oranges
            ("GEN-120", "Light Orange",   (255, 190, 92)),
            ("GEN-121", "Orange",         (255, 140, 0)),
            ("GEN-122", "Burnt Orange",   (204, 102, 0)),
            # browns (good for animals)
            ("GEN-140", "Light Tan",      (230, 205, 170)),
            ("GEN-141", "Tan",            (210, 180, 140)),
            ("GEN-142", "Camel",          (193, 145, 80)),
            ("GEN-143", "Bronze",         (150, 100, 40)),
            ("GEN-144", "Brown",          (139, 69, 19)),
            ("GEN-145", "Dark Brown",     (101, 67, 33)),
            ("GEN-146", "Umber",          (78, 53, 36)),
            # reds
            ("GEN-200", "Light Red",      (255, 102, 102)),
            ("GEN-201", "Red",            (220, 20, 60)),
            ("GEN-202", "Maroon",         (128, 0, 0)),
            # pinks
            ("GEN-220", "Rose",           (255, 105, 180)),
            ("GEN-221", "Light Pink",     (255, 182, 193)),
            # purples
            ("GEN-240", "Lavender",       (182, 145, 225)),
            ("GEN-241", "Purple",         (128, 0, 128)),
            # blues
            ("GEN-300", "Sky Blue",       (135, 206, 235)),
            ("GEN-301", "Azure",          (80, 170, 255)),
            ("GEN-302", "Royal Blue",     (65, 105, 225)),
            ("GEN-303", "Blue",           (0, 102, 204)),
            ("GEN-304", "Navy",           (0, 0, 128)),
            # teals
            ("GEN-320", "Turquoise",      (64, 224, 208)),
            ("GEN-321", "Teal",           (0, 128, 128)),
            # greens
            ("GEN-340", "Lime",           (50, 205, 50)),
            ("GEN-341", "Green",          (0, 128, 0)),
            ("GEN-342", "Forest",         (0, 90, 0)),
            ("GEN-343", "Olive",          (107, 142, 35)),
            # creams / beiges
            ("GEN-360", "Ivory",          (255, 250, 240)),
            ("GEN-361", "Cream",          (255, 253, 208)),
            ("GEN-362", "Beige",          (245, 245, 220)),
            ("GEN-363", "Warm Gray",      (150, 120, 110)),
            ("GEN-364", "Cool Gray",      (120, 140, 160)),
        ]
    },
    "Madeira Classic 40 (mini)": {
        "desc": "Madeira Classic 40 – compact subset tuned for logos & animals",
        "colors": [
            ("MA-1000", "Snow White",       (255, 255, 255)),
            ("MA-1005", "Ivory",            (250, 246, 230)),
            ("MA-1061", "Pale Yellow",      (255, 233, 128)),
            ("MA-1072", "Lemon",            (255, 242, 0)),
            ("MA-1078", "Yellow",           (255, 210, 0)),
            ("MA-1114", "Gold",             (204, 153, 0)),
            ("MA-1147", "Orange",           (255, 140, 0)),
            ("MA-1183", "Burnt Orange",     (198, 108, 58)),
            ("MA-1222", "Beige",            (232, 216, 186)),
            ("MA-1243", "Light Tan",        (214, 184, 141)),
            ("MA-1267", "Tan",              (196, 164, 124)),
            ("MA-1305", "Camel",            (190, 146, 93)),
            ("MA-1311", "Brown",            (145, 100, 60)),
            ("MA-1334", "Dark Brown",       (102, 71, 41)),
            ("MA-1398", "Chocolate",        (76, 50, 32)),
            ("MA-1500", "Red",              (220, 30, 40)),
            ("MA-1624", "Magenta",          (186, 80, 160)),
            ("MA-1755", "Lavender",         (176, 148, 209)),
            ("MA-1805", "Sky Blue",         (146, 206, 235)),
            ("MA-1842", "Royal Blue",       (54, 90, 210)),
            ("MA-1860", "Navy",             (0, 36, 102)),
            ("MA-1952", "Emerald",          (0, 160, 98)),
            ("MA-1960", "Green",            (0, 128, 0)),
            ("MA-1999", "Black",            (0, 0, 0)),
        ]
    },
}

def map_palette_to_brand(palette_rgb: List[RGB], brand: str = "Generic 40wt (basic)") -> List[dict]:
    """Return a list (same length as palette_rgb) of dicts with nearest thread info."""
    brand = brand if brand in THREAD_BRANDS else "Generic 40wt (basic)"
    lib = THREAD_BRANDS[brand]["colors"]
    lib_rgb = [c[2] for c in lib]
    lib_lab = [_rgb_to_lab(c) for c in lib_rgb]

    out = []
    for rgb in palette_rgb:
        lab = _rgb_to_lab(rgb)
        # find nearest by ΔE76
        best_i, best_d = 0, 1e9
        for i, lab2 in enumerate(lib_lab):
            d = _de76(lab, lab2)
            if d < best_d:
                best_d, best_i = d, i
        code, name, libc = lib[best_i]
        out.append({
            "brand": brand,
            "code": code,
            "name": name,
            "rgb": libc,
            "hex": _hex(libc),
            "distance": float(best_d),
            "source_rgb": tuple(int(v) for v in rgb),
        })
    return out
