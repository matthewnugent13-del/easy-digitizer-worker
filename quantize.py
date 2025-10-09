# quantize.py
# Palette quantization that ONLY ignores transparency.
# True whites (or any light color) remain stitchable.

from typing import List, Tuple
import numpy as np
from PIL import Image

BACKGROUND_INDEX = 255  # reserved index for non-stitchable background (transparent)

# Try scikit-learn KMeans if available; otherwise fall back to Pillow adaptive quantization
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def _image_to_rgba(img: Image.Image) -> Image.Image:
    # Always work in RGBA so we can read the alpha channel
    if img.mode == "RGBA":
        return img
    if img.mode in ("RGB", "P", "L", "LA"):
        return img.convert("RGBA")
    return img.convert("RGBA")


def _quantize_with_kmeans(rgb: np.ndarray, mask: np.ndarray, n_colors: int, random_state: int = 42):
    """
    rgb: HxWx3 uint8
    mask: HxW bool (True = pixel participates)
    returns: (indexed HxW uint8, palette Nx3 uint8)
    """
    H, W, _ = rgb.shape
    indexed = np.full((H, W), BACKGROUND_INDEX, dtype=np.uint8)

    # Guard: if there are no opaque pixels, return empty palette and all background
    if not np.any(mask):
        return indexed, np.zeros((0, 3), dtype=np.uint8)

    # Prepare samples
    samples = rgb[mask].astype(np.float32)
    # Limit clusters to available unique colors
    uniq = np.unique(samples.astype(np.uint8), axis=0)
    k = int(max(1, min(n_colors, len(uniq))))

    km = KMeans(n_clusters=k, n_init=4, random_state=random_state)
    km.fit(samples)
    centers = np.clip(km.cluster_centers_.astype(np.int32), 0, 255).astype(np.uint8)

    # Predict labels for all masked pixels
    labels = km.predict(samples)
    flat = indexed.ravel()
    idxs = np.flatnonzero(mask.ravel())
    flat[idxs] = labels.astype(np.uint8)

    return indexed, centers


def _quantize_with_pillow(img_rgba: Image.Image, n_colors: int):
    """
    Fallback: use Pillow's adaptive quantizer on the whole image,
    then re-apply transparency mask so transparent pixels become BACKGROUND_INDEX.
    """
    H, W = img_rgba.size[1], img_rgba.size[0]
    arr = np.array(img_rgba, dtype=np.uint8)
    alpha = arr[:, :, 3]
    opaque = alpha >= 8
    # Use Adaptive quantization to at most n_colors (keeps whites if present)
    pal_img = img_rgba.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=max(1, n_colors))
    # Pull back to indexed + palette
    pal = pal_img.getpalette()[: 3 * 256]
    pal = np.array(pal, dtype=np.uint8).reshape(-1, 3)[:n_colors]  # top n colors
    idx = np.array(pal_img, dtype=np.uint8)
    # Map any index >= n_colors down (rare)
    idx[idx >= n_colors] = (n_colors - 1) if n_colors > 0 else 0
    # Put BACKGROUND_INDEX where transparent
    out = np.where(opaque, idx, BACKGROUND_INDEX).astype(np.uint8)
    return out, pal


def quantize_image(
    img: Image.Image,
    n_colors: int = 6,
    remove_bg: bool = True,
    ignore_alpha_only: bool = True,
    alpha_threshold: int = 8,
):
    """
    Quantize image to up to n_colors, keeping whites. The ONLY pixels dropped to BACKGROUND_INDEX
    are those with alpha < alpha_threshold WHEN remove_bg and ignore_alpha_only are True.

    Returns:
        indexed (HxW uint8), palette_rgb (List[Tuple[int,int,int]]), preview_img (PIL RGB)
    """
    img_rgba = _image_to_rgba(img)
    arr = np.array(img_rgba, dtype=np.uint8)  # HxWx4
    H, W, _ = arr.shape
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    if remove_bg and ignore_alpha_only:
        opaque = alpha >= alpha_threshold
    else:
        # Treat everything as stitchable (no background removal)
        opaque = np.ones((H, W), dtype=bool)

    if HAS_SKLEARN:
        indexed, palette = _quantize_with_kmeans(rgb, opaque, n_colors)
    else:
        indexed, palette = _quantize_with_pillow(img_rgba, n_colors)

    # Build a preview image
    preview = np.zeros((H, W, 3), dtype=np.uint8)
    if palette.size > 0:
        # Flatten for fast assignment
        pal = palette.astype(np.uint8)
        valid = indexed != BACKGROUND_INDEX
        preview[valid] = pal[indexed[valid]]
    # For transparent pixels, leave black (or you could render a checkerboard)
    preview_img = Image.fromarray(preview, mode="RGB")

    palette_list: List[Tuple[int, int, int]] = [tuple(int(c) for c in row) for row in palette]
    return indexed, palette_list, preview_img
