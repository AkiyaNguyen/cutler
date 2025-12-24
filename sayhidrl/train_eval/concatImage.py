from PIL import Image, ImageDraw, ImageFont
import os

from PIL import Image
import os
from PIL import Image
import os

def make_1xN(image_paths, save_path, pad=10, target_size=None):
    """
    image_paths: list of image paths
    target_size: (W, H) or None
    """

    imgs = []

    # ---- Load images
    for p in image_paths:
        imgs.append(Image.open(p).convert("RGB"))

    # ---- Determine target size
    if target_size is None:
        target_size = imgs[0].size  # (W, H)

    # ---- Resize all images
    imgs = [img.resize(target_size, Image.BILINEAR) for img in imgs]

    w, h = target_size
    N = len(imgs)

    # ---- Create canvas
    canvas = Image.new(
        "RGB",
        (N * w + (N - 1) * pad, h),
        (0, 0, 0)
    )

    # ---- Paste images
    for i in range(N):
        x = i * (w + pad)
        canvas.paste(imgs[i], (x, 0))

    canvas.save(save_path)
    print(f"Saved horizontal concatenation to {save_path}")

def make_2xN(samples, save_path, pad=10, target_size=None):
    """
    samples: list of tuple (row1_img_path, row2_img_path)
    target_size: (W, H) or None
    """

    imgs_row1, imgs_row2 = [], []

    # ---- Load images
    for p1, p2 in samples:
        imgs_row1.append(Image.open(p1).convert("RGB"))
        imgs_row2.append(Image.open(p2).convert("RGB"))

    # ---- Determine target size
    if target_size is None:
        target_size = imgs_row1[0].size  # (W, H)

    # ---- Resize all images
    imgs_row1 = [img.resize(target_size, Image.BILINEAR) for img in imgs_row1]
    imgs_row2 = [img.resize(target_size, Image.BILINEAR) for img in imgs_row2]

    w, h = target_size
    N = len(samples)

    # ---- Create canvas
    canvas = Image.new(
        "RGB",
        (N * w + (N - 1) * pad, 2 * h + pad),
        (0, 0, 0)
    )

    # ---- Paste images
    for i in range(N):
        x = i * (w + pad)
        canvas.paste(imgs_row1[i], (x, 0))
        canvas.paste(imgs_row2[i], (x, h + pad))

    canvas.save(save_path)
    print(f"Saved qualitative comparison to {save_path}")

    
images = [
    "evaluation/original/img_2.png",
    "evaluation/gt/mask_2.png",
    "evaluation/raw_cascade_003/mask/mask_2.png",
    "evaluation/cutler_cascade_003/mask/mask_2.png",
]

make_1xN(
    images,
    "qualitative_1xN.png",
    target_size=(512, 512)
)