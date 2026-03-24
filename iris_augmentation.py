"""
Advanced Iris Image Dataset Augmentation Script
================================================
Generates 10,000 augmented iris images (5000 healthy + 5000 abnormal)
from a small original dataset using OpenCV, NumPy, and TensorFlow utilities.

Input  : ./original_dataset/   (folder containing source iris images)
Output : ./healthy_augmented/   (5000 images)
         ./abnormal_augmented/  (5000 images)

Usage  : python iris_augmentation.py
"""

import cv2
import numpy as np
import os
import random
import math
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INPUT_DIR        = "original_dataset"
HEALTHY_DIR      = "healthy_augmented"
ABNORMAL_DIR     = "abnormal_augmented"
TARGET_SIZE      = (128, 128)        # Resize all images to 128×128
TARGET_HEALTHY   = 5000
TARGET_ABNORMAL  = 5000
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
PROGRESS_STEP    = 100               # Print update every N images


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def prepare_dirs():
    """Create output directories if they don't exist."""
    for d in [HEALTHY_DIR, ABNORMAL_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_images(folder: str) -> list[np.ndarray]:
    """
    Load all supported images from the input folder,
    resize them to TARGET_SIZE and return as a list of BGR arrays.
    Falls back to a synthetic iris if the folder is empty or missing.
    """
    images = []
    if not os.path.isdir(folder):
        print(f"[WARNING] '{folder}' not found – generating synthetic seed images.")
        return images

    for fname in os.listdir(folder):
        ext = Path(fname).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] Cannot read: {fname}")
            continue
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        images.append(img)

    print(f"[INFO] Loaded {len(images)} source image(s) from '{folder}'.")
    return images


def generate_synthetic_iris() -> np.ndarray:
    """
    Create a synthetic 128×128 iris-like image when no real
    source images are available. Produces a realistic radial texture.
    """
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cx, cy = 64, 64

    # Sclera (white of eye)
    cv2.circle(img, (cx, cy), 62, (220, 215, 210), -1)

    # Base iris colour (brownish-hazel)
    base_color = (random.randint(50, 100), random.randint(80, 130), random.randint(100, 160))
    cv2.circle(img, (cx, cy), 42, base_color, -1)

    # Radial fibre lines
    for angle in range(0, 360, 3):
        rad = math.radians(angle)
        r = random.randint(22, 40)
        x2 = int(cx + r * math.cos(rad))
        y2 = int(cy + r * math.sin(rad))
        intensity = random.randint(-20, 20)
        line_color = tuple(min(255, max(0, c + intensity)) for c in base_color)
        cv2.line(img, (cx, cy), (x2, y2), line_color, 1)

    # Pupil (dark centre)
    cv2.circle(img, (cx, cy), 18, (15, 10, 10), -1)

    # Specular highlight
    cv2.circle(img, (cx + 6, cy - 6), 4, (240, 240, 240), -1)

    # Limbal ring (dark edge around iris)
    cv2.circle(img, (cx, cy), 42, (30, 25, 20), 2)

    return img


# ─────────────────────────────────────────────
# General augmentation (applied to ALL classes)
# ─────────────────────────────────────────────

def augment_rotation(img: np.ndarray) -> np.ndarray:
    """Rotate image randomly between -30° and +30°."""
    angle = random.uniform(-30, 30)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def augment_brightness(img: np.ndarray) -> np.ndarray:
    """Randomly increase or decrease overall brightness."""
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    delta = random.uniform(0.5, 1.6)          # factor applied to V channel
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * delta, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_contrast(img: np.ndarray) -> np.ndarray:
    """Adjust image contrast using alpha scaling around the mid-point."""
    alpha = random.uniform(0.6, 1.8)          # contrast factor
    beta  = random.uniform(-30, 30)           # brightness shift
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def augment_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Apply mild Gaussian blur to simulate slight defocus."""
    ksize = random.choice([3, 5, 7])          # kernel must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def augment_gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise."""
    sigma = random.uniform(5, 25)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_zoom(img: np.ndarray) -> np.ndarray:
    """
    Zoom in (crop centre) or zoom out (pad + resize).
    Zoom factor between 0.75× (out) and 1.3× (in).
    """
    h, w  = img.shape[:2]
    scale = random.uniform(0.75, 1.30)
    new_h = int(h * scale)
    new_w = int(w * scale)

    if scale > 1.0:
        # Zoom in: resize bigger then centre-crop back to original size
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        y1 = (new_h - h) // 2
        x1 = (new_w - w) // 2
        return resized[y1:y1 + h, x1:x1 + w]
    else:
        # Zoom out: resize smaller then pad back to original size
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_y   = (h - new_h) // 2
        pad_x   = (w - new_w) // 2
        padded  = cv2.copyMakeBorder(
            resized,
            pad_y, h - new_h - pad_y,
            pad_x, w - new_w - pad_x,
            cv2.BORDER_REFLECT_101,
        )
        return cv2.resize(padded, (w, h))


def augment_shift(img: np.ndarray) -> np.ndarray:
    """
    Apply a random horizontal and vertical translation.
    Shift amount is ±15% of the image dimension.
    """
    h, w  = img.shape[:2]
    tx    = random.uniform(-0.15, 0.15) * w   # horizontal shift
    ty    = random.uniform(-0.15, 0.15) * h   # vertical shift
    M     = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def apply_general_augmentations(img: np.ndarray) -> np.ndarray:
    """
    Randomly apply a subset of general augmentations.
    Each transformation has an independent probability of being applied.
    """
    ops = [
        (0.7, augment_rotation),
        (0.6, augment_brightness),
        (0.6, augment_contrast),
        (0.4, augment_gaussian_blur),
        (0.5, augment_gaussian_noise),
        (0.5, augment_zoom),
        (0.5, augment_shift),
    ]
    for prob, fn in ops:
        if random.random() < prob:
            img = fn(img)
    return img


# ─────────────────────────────────────────────
# Abnormal-specific augmentations
# ─────────────────────────────────────────────

def abnormal_dark_spots(img: np.ndarray) -> np.ndarray:
    """
    Add 1–4 artificial dark spots on the iris region to mimic
    pathological deposits or melanoma-like pigment accumulations.
    """
    out  = img.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    for _ in range(random.randint(1, 4)):
        # Restrict spots to the iris ring (radius 18–42 px from centre)
        angle  = random.uniform(0, 2 * math.pi)
        radius = random.uniform(18, 40)
        sx     = int(cx + radius * math.cos(angle))
        sy     = int(cy + radius * math.sin(angle))
        sr     = random.randint(2, 6)            # spot radius
        alpha  = random.uniform(0.3, 0.7)        # blend factor (0=invisible, 1=solid)

        # Draw filled dark ellipse (slightly irregular shape)
        axes  = (sr, max(1, sr + random.randint(-2, 2)))
        rot_a = random.randint(0, 180)
        mask  = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (sx, sy), axes, rot_a, 0, 360, 255, -1)

        dark_color = np.array([random.randint(5, 30)] * 3, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = np.where(
                mask > 0,
                np.clip(out[:, :, c] * (1 - alpha) + dark_color[c] * alpha, 0, 255),
                out[:, :, c],
            ).astype(np.uint8)

    return out


def abnormal_pigmentation_patch(img: np.ndarray) -> np.ndarray:
    """
    Add a reddish-brown pigmentation patch onto the iris,
    simulating heterochromia iridum or sectoral pigment anomalies.
    """
    out  = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    angle  = random.uniform(0, 2 * math.pi)
    radius = random.uniform(15, 35)
    px     = int(cx + radius * math.cos(angle))
    py     = int(cy + radius * math.sin(angle))

    # Wedge-shaped region
    patch_size = random.randint(10, 22)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (px, py),
                (patch_size, patch_size // 2),
                random.randint(0, 180), 0, 360, 1.0, -1)

    # Blur mask edges for a natural look
    mask = cv2.GaussianBlur(mask, (11, 11), 4)

    pigment = np.array([
        random.randint(20, 60),    # B – low blue
        random.randint(40, 90),    # G – moderate green
        random.randint(90, 150),   # R – warm red-brown
    ], dtype=np.float32)

    strength = random.uniform(0.3, 0.6)
    for c in range(3):
        out[:, :, c] = np.clip(
            out[:, :, c] * (1 - mask * strength) + pigment[c] * (mask * strength),
            0, 255,
        )

    return out.astype(np.uint8)


def abnormal_texture_distortion(img: np.ndarray) -> np.ndarray:
    """
    Apply slight elastic-style distortion to simulate fibrous
    texture irregularities (e.g., stromal thinning or crypts).
    """
    h, w = img.shape[:2]
    # Random displacement maps
    dx = np.random.uniform(-4, 4, (h, w)).astype(np.float32)
    dy = np.random.uniform(-4, 4, (h, w)).astype(np.float32)

    # Smooth so distortion is gradual, not per-pixel noise
    dx = cv2.GaussianBlur(dx, (15, 15), 5)
    dy = cv2.GaussianBlur(dy, (15, 15), 5)

    # Build remapping coordinates
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def abnormal_localized_shadow(img: np.ndarray) -> np.ndarray:
    """
    Cast a soft shadow over part of the iris to simulate
    vascular engorgement, neovascularisation, or recording artefacts.
    """
    out  = img.copy().astype(np.float32)
    h, w = img.shape[:2]

    # Random semi-transparent dark wedge originating near the pupil edge
    mask     = np.zeros((h, w), dtype=np.float32)
    angle    = random.uniform(0, 360)
    span     = random.uniform(30, 90)     # angular width of the shadow
    cx, cy   = w // 2, h // 2
    max_r    = 45

    # Filled polygon approximating a wedge
    pts = [(cx, cy)]
    for a in range(int(angle), int(angle + span), 4):
        rad = math.radians(a)
        pts.append((int(cx + max_r * math.cos(rad)),
                    int(cy + max_r * math.sin(rad))))
    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_arr], 1.0)
    mask = cv2.GaussianBlur(mask, (21, 21), 8)

    dark = random.uniform(0.35, 0.65)     # how dark the shadow is
    out  = out * (1 - mask[:, :, np.newaxis] * dark)

    return np.clip(out, 0, 255).astype(np.uint8)


def abnormal_circular_disruption(img: np.ndarray) -> np.ndarray:
    """
    Draw faint irregular concentric ring distortions to simulate
    iris sphincter atrophy or collarette abnormalities.
    """
    out  = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    num_rings = random.randint(1, 3)
    for _ in range(num_rings):
        r      = random.randint(20, 42)
        thick  = random.randint(1, 3)
        color  = np.array([random.randint(20, 80)] * 3, dtype=np.float32)
        alpha  = random.uniform(0.2, 0.5)

        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), r, 255, thick)

        for c in range(3):
            out[:, :, c] = np.where(
                ring_mask > 0,
                np.clip(out[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255),
                out[:, :, c],
            )

    return out.astype(np.uint8)


def apply_abnormal_augmentations(img: np.ndarray) -> np.ndarray:
    """
    Apply 2–5 randomly chosen abnormal transformations on top of
    the general augmentations to produce a clinically plausible
    pathological iris appearance.
    """
    abnormal_ops = [
        abnormal_dark_spots,
        abnormal_pigmentation_patch,
        abnormal_texture_distortion,
        abnormal_localized_shadow,
        abnormal_circular_disruption,
    ]
    # Choose a random subset (at least 2) of the abnormal ops
    k        = random.randint(2, len(abnormal_ops))
    chosen   = random.sample(abnormal_ops, k)
    for fn in chosen:
        img = fn(img)
    return img


# ─────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────

def generate_dataset(source_images: list[np.ndarray]):
    """
    Main loop: generates TARGET_HEALTHY healthy images and
    TARGET_ABNORMAL abnormal images, saving them with zero-padded
    sequential filenames (e.g. iris_aug_00001.jpg).
    """
    healthy_count  = 0
    abnormal_count = 0
    total_target   = TARGET_HEALTHY + TARGET_ABNORMAL
    global_counter = 0   # used for unique filenames across both classes

    print(f"\n{'='*60}")
    print(f"  CKD Iris Augmentation — Target: {total_target} images")
    print(f"  Healthy : {TARGET_HEALTHY}   |   Abnormal : {TARGET_ABNORMAL}")
    print(f"{'='*60}\n")

    # Cycle through source images indefinitely until targets are met
    source_cycle = source_images[:]

    while healthy_count < TARGET_HEALTHY or abnormal_count < TARGET_ABNORMAL:
        # Pick a random source image (with repetition to handle tiny datasets)
        base_img = random.choice(source_cycle)

        # ── Decide which class to generate ──────────────────────────
        # Randomly pick a class that still needs more images
        remaining_h = TARGET_HEALTHY  - healthy_count
        remaining_a = TARGET_ABNORMAL - abnormal_count

        if remaining_h <= 0:
            gen_class = "abnormal"
        elif remaining_a <= 0:
            gen_class = "healthy"
        else:
            gen_class = "healthy" if random.random() < 0.5 else "abnormal"

        # ── Apply augmentations ──────────────────────────────────────
        aug_img = apply_general_augmentations(base_img.copy())

        if gen_class == "abnormal":
            aug_img = apply_abnormal_augmentations(aug_img)

        # ── Save image ───────────────────────────────────────────────
        global_counter += 1
        filename = f"iris_aug_{global_counter:05d}.jpg"

        if gen_class == "healthy":
            save_path = os.path.join(HEALTHY_DIR, filename)
            healthy_count += 1
        else:
            save_path = os.path.join(ABNORMAL_DIR, filename)
            abnormal_count += 1

        cv2.imwrite(save_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── Progress reporting ───────────────────────────────────────
        total_done = healthy_count + abnormal_count
        if total_done % PROGRESS_STEP == 0 or total_done == total_target:
            pct = (total_done / total_target) * 100
            bar_len   = 30
            filled    = int(bar_len * total_done / total_target)
            bar       = "█" * filled + "░" * (bar_len - filled)
            print(
                f"  [{bar}] {pct:5.1f}%  "
                f"Healthy: {healthy_count:5d}/{TARGET_HEALTHY}  "
                f"Abnormal: {abnormal_count:5d}/{TARGET_ABNORMAL}  "
                f"Total: {total_done:5d}/{total_target}",
                end="\r",
                flush=True,
            )

    print()   # newline after the progress bar
    print(f"\n{'='*60}")
    print(f"  Done! Generated {healthy_count + abnormal_count} images.")
    print(f"  Healthy  saved to : ./{HEALTHY_DIR}/")
    print(f"  Abnormal saved to : ./{ABNORMAL_DIR}/")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Prepare output directories
    prepare_dirs()

    # ── 2. Load source images
    source_images = load_images(INPUT_DIR)

    # ── 3. If no real images found, generate synthetic seeds
    if not source_images:
        print("[INFO] No source images found – creating 20 synthetic iris seeds.")
        source_images = [generate_synthetic_iris() for _ in range(20)]

    # ── 4. Generate the augmented dataset
    generate_dataset(source_images)
