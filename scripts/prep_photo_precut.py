"""
Variant of prep_photo.py for photos that ALREADY have the background removed
(i.e. already have a real alpha channel) -- skips rembg entirely and just does
the CLAHE local-contrast pass + white composite.

    python scripts/prep_photo_precut.py <input.png> [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

cut = Image.open(INP).convert("RGBA")
rgb = np.array(cut.convert("RGB"))
alpha = np.array(cut.split()[-1])  # 0 = background

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
gray = clahe.apply(gray)

gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18)

mask = (alpha.astype(np.float32) / 255.0)
mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
out = np.clip(out, 0, 255).astype(np.uint8)

Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape)
