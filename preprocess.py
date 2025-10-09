# preprocess.py
from PIL import Image

def fit_to_hoop(image: Image.Image, hoop_mm=(130, 180), px_per_mm=10):
    hoop_w_px = int(hoop_mm[0] * px_per_mm)
    hoop_h_px = int(hoop_mm[1] * px_per_mm)

    # keep alpha if present
    im = image.convert("RGBA")

    # resize with aspect ratio
    img_w, img_h = im.size
    scale = min(hoop_w_px / img_w, hoop_h_px / img_h)
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))
    resized = im.resize((new_w, new_h), Image.LANCZOS)

    # make a TRANSPARENT canvas (not white!)
    canvas = Image.new("RGBA", (hoop_w_px, hoop_h_px), (0, 0, 0, 0))
    offset_x = (hoop_w_px - new_w) // 2
    offset_y = (hoop_h_px - new_h) // 2
    canvas.alpha_composite(resized, (offset_x, offset_y))
    return canvas
