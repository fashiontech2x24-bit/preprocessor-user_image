"""
SAM3 Garment Segmentation API Server
=====================================
FastAPI server that uses Meta's SAM 3 (Segment Anything Model 3)
for text-prompted garment segmentation in a VTON pipeline.

Input: user_image + type
Output: masked_image (green-tinted blur on segmented garment area)
"""

import os
import io
import time
import logging
from enum import Enum
from typing import Optional

import torch
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sam3-vton")

# ──────────────────────────── Config ─────────────────────────────

# Default text prompts per garment type – editable at runtime via /config
DEFAULT_PROMPTS = {
    "full":    "clothing on the person",
    "upper":   "upper body garment shirt top blouse",
    "lower":   "lower body garment pants trousers skirt",
    "layered": "outer wear jacket coat hoodie blazer",
}

# Mask overlay config – tunable via /config
MASK_CONFIG = {
    "green_color": (0, 255, 0),       # RGB green overlay
    "green_alpha": 0.45,              # opacity of green tint (0-1)
    "blur_radius": 100,               # max gaussian blur radius
    "border_width": 3,                # clear border width in pixels (thin)
    "threshold": 0.5,                 # SAM3 detection threshold
    "mask_threshold": 0.5,            # SAM3 mask binarization threshold
}

# ──────────────────────────── App ────────────────────────────────
app = FastAPI(
    title="SAM3 Garment Segmentation API",
    description="VTON preprocessing: segment garments with SAM 3 text prompts → green-tinted blur mask",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────── Global State ───────────────────────
model = None
processor = None
device = None
prompts = dict(DEFAULT_PROMPTS)
mask_config = dict(MASK_CONFIG)


class GarmentType(str, Enum):
    full = "full"
    upper = "upper"
    lower = "lower"
    layered = "layered"


# ──────────────────────────── Model Loading ──────────────────────
def load_model():
    """Load SAM 3 model and processor via official sam3 package."""
    global model, processor, device

    logger.info("Loading SAM 3 model...")
    start = time.time()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor as Sam3Proc

    model = build_sam3_image_model()
    processor = Sam3Proc(model)
    logger.info(f"SAM 3 loaded in {time.time() - start:.1f}s")


# ──────────────────────────── Preprocessing ──────────────────────

CANVAS_W, CANVAS_H = 768, 1024


def preprocess_to_canvas(image: Image.Image) -> Image.Image:
    """
    Fit image into a 768×1024 canvas with reflect-fill padding.

    Steps:
        1. Compute scale factor to fit within canvas (preserve aspect ratio)
        2. Resize with LANCZOS (best quality for downscale)
        3. Paste centered onto canvas
        4. Fill empty bands with reflected/mirrored strips from image edges
    """
    orig_w, orig_h = image.size

    # Already exact size — skip
    if orig_w == CANVAS_W and orig_h == CANVAS_H:
        return image.copy()

    # 1. Scale to fit within canvas
    scale = min(CANVAS_W / orig_w, CANVAS_H / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    logger.info(
        f"Preprocess: {orig_w}×{orig_h} → scale {scale:.3f} → "
        f"{new_w}×{new_h} → canvas {CANVAS_W}×{CANVAS_H}"
    )

    # 2. Paste centered
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    offset_x = (CANVAS_W - new_w) // 2
    offset_y = (CANVAS_H - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    # 3. Reflect-fill empty bands
    canvas_arr = np.array(canvas)
    resized_arr = np.array(resized)

    # ── Top band ──
    if offset_y > 0:
        # Mirror the top rows of the resized image
        strip_h = min(offset_y, new_h)
        strip = resized_arr[:strip_h, :, :]
        reflected = strip[::-1, :, :]  # vertical flip
        # Tile if strip is smaller than gap
        fill = _tile_reflect(reflected, offset_y, axis=0)
        canvas_arr[:offset_y, offset_x:offset_x + new_w, :] = fill[:, :new_w, :]

    # ── Bottom band ──
    bottom_start = offset_y + new_h
    bottom_gap = CANVAS_H - bottom_start
    if bottom_gap > 0:
        strip_h = min(bottom_gap, new_h)
        strip = resized_arr[-strip_h:, :, :]
        reflected = strip[::-1, :, :]
        fill = _tile_reflect(reflected, bottom_gap, axis=0)
        canvas_arr[bottom_start:, offset_x:offset_x + new_w, :] = fill[:, :new_w, :]

    # ── Left band (full height now) ──
    if offset_x > 0:
        strip_w = min(offset_x, new_w)
        strip = canvas_arr[:, offset_x:offset_x + strip_w, :]
        reflected = strip[:, ::-1, :]
        fill = _tile_reflect(reflected, offset_x, axis=1)
        canvas_arr[:, :offset_x, :] = fill

    # ── Right band (full height now) ──
    right_start = offset_x + new_w
    right_gap = CANVAS_W - right_start
    if right_gap > 0:
        strip_w = min(right_gap, new_w)
        strip = canvas_arr[:, right_start - strip_w:right_start, :]
        reflected = strip[:, ::-1, :]
        fill = _tile_reflect(reflected, right_gap, axis=1)
        canvas_arr[:, right_start:, :] = fill

    return Image.fromarray(canvas_arr)


def _tile_reflect(strip: np.ndarray, target_size: int, axis: int) -> np.ndarray:
    """Tile a reflected strip to fill target_size along the given axis."""
    current = strip.shape[axis]
    if current >= target_size:
        slices = [slice(None)] * strip.ndim
        slices[axis] = slice(0, target_size)
        return strip[tuple(slices)]

    # Need to repeat — alternate reflect direction
    pieces = [strip]
    filled = current
    flip = True
    while filled < target_size:
        piece = np.flip(pieces[-1], axis=axis) if flip else pieces[-1].copy()
        pieces.append(piece)
        filled += piece.shape[axis]
        flip = not flip

    joined = np.concatenate(pieces, axis=axis)
    slices = [slice(None)] * joined.ndim
    slices[axis] = slice(0, target_size)
    return joined[tuple(slices)]


# ──────────────────────────── Segmentation ───────────────────────

def segment_image(image: Image.Image, text_prompt: str) -> np.ndarray:
    """Segment using the official sam3 package API."""
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=text_prompt,
    )
    masks = output["masks"]       # list of binary masks
    scores = output["scores"]     # confidence scores

    if len(masks) == 0:
        logger.warning(f"No segments found for prompt: '{text_prompt}'")
        return np.zeros((image.height, image.width), dtype=bool)

    # Combine all detected masks into a single union mask
    combined = np.zeros((image.height, image.width), dtype=bool)
    for m, s in zip(masks, scores):
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        # Resize mask to image dimensions if needed
        if m.shape != (image.height, image.width):
            from PIL import Image as PILImage
            m_pil = PILImage.fromarray(m.astype(np.uint8) * 255)
            m_pil = m_pil.resize((image.width, image.height), PILImage.NEAREST)
            m = np.array(m_pil) > 127
        combined = combined | m.astype(bool)

    logger.info(f"Segmented {len(masks)} instances for prompt: '{text_prompt}'")
    return combined


def apply_green_blur_mask(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """
    Apply green-tinted gaussian blur on the segmented area
    with a thin clear border around the mask edge.

    Flow:
    1. Heavy gaussian blur on full image
    2. Green tint the blurred region
    3. Erode the mask slightly to create a clear border
    4. Composite: original outside mask, green-blur inside eroded mask,
       clear original border between the two
    """
    cfg = mask_config
    green_color = cfg["green_color"]
    green_alpha = cfg["green_alpha"]
    blur_radius = cfg["blur_radius"]
    border_width = cfg["border_width"]

    img_array = np.array(image).copy()
    h, w = img_array.shape[:2]

    # 1. Create heavily blurred version
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_array = np.array(blurred)

    # 2. Apply green tint to blurred region
    green_overlay = np.full_like(img_array, green_color, dtype=np.uint8)
    green_blurred = (
        blurred_array.astype(np.float32) * (1 - green_alpha)
        + green_overlay.astype(np.float32) * green_alpha
    ).astype(np.uint8)

    # 3. Erode mask to create inner region (border = mask - eroded_mask)
    from scipy.ndimage import binary_erosion
    if border_width > 0:
        eroded_mask = binary_erosion(mask, iterations=border_width)
    else:
        eroded_mask = mask.copy()

    # 4. Composite
    # - Outside mask: original image
    # - Border region (mask but not eroded): original (clear border)
    # - Inside eroded mask: green-blurred
    result = img_array.copy()
    result[eroded_mask] = green_blurred[eroded_mask]

    return Image.fromarray(result)


# ──────────────────────────── Startup ────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()


# ──────────────────────────── Endpoints ──────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "prompts": prompts,
        "mask_config": mask_config,
    }


@app.post("/segment")
async def segment(
    user_image: UploadFile = File(..., description="User photo to segment"),
    type: GarmentType = Form(..., description="Garment type: full, upper, lower, layered"),
    prompt_override: Optional[str] = Form(None, description="Override the default text prompt"),
):
    """
    Main endpoint: segment garment from user image.

    Returns the processed image with green-tinted blur applied
    to the segmented garment area.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()

    # Read and validate image
    try:
        contents = await user_image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Preprocess: fit to 768×1024 canvas with reflect padding
    image = preprocess_to_canvas(image)

    # Determine prompt
    text_prompt = prompt_override if prompt_override else prompts[type.value]
    logger.info(f"Segmenting type='{type.value}' with prompt='{text_prompt}' | image={image.size}")

    # Segment
    try:
        mask = segment_image(image, text_prompt)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    mask_coverage = mask.sum() / mask.size * 100
    logger.info(f"Mask coverage: {mask_coverage:.1f}%")

    if mask_coverage < 0.1:
        logger.warning("Mask coverage extremely low — prompt may need adjustment")

    # Apply green blur mask
    try:
        result = apply_green_blur_mask(image, mask)
    except Exception as e:
        logger.error(f"Mask application failed: {e}")
        raise HTTPException(status_code=500, detail=f"Mask application failed: {e}")

    # Encode result
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.2f}s")

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Processing-Time": f"{elapsed:.3f}",
            "X-Mask-Coverage": f"{mask_coverage:.1f}",
            "X-Prompt-Used": text_prompt,
        },
    )


@app.post("/segment/raw-mask")
async def segment_raw_mask(
    user_image: UploadFile = File(..., description="User photo to segment"),
    type: GarmentType = Form(..., description="Garment type: full, upper, lower, layered"),
    prompt_override: Optional[str] = Form(None, description="Override the default text prompt"),
):
    """
    Returns the raw binary mask as a PNG (white = garment, black = background).
    Useful for debugging prompts.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        contents = await user_image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Preprocess: fit to 768×1024 canvas with reflect padding
    image = preprocess_to_canvas(image)

    text_prompt = prompt_override if prompt_override else prompts[type.value]

    try:
        mask = segment_image(image, text_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # Convert mask to image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


class PromptUpdate(BaseModel):
    full: Optional[str] = None
    upper: Optional[str] = None
    lower: Optional[str] = None
    layered: Optional[str] = None


class MaskConfigUpdate(BaseModel):
    green_color: Optional[list] = None
    green_alpha: Optional[float] = None
    blur_radius: Optional[int] = None
    border_width: Optional[int] = None
    threshold: Optional[float] = None
    mask_threshold: Optional[float] = None


@app.post("/config/prompts")
async def update_prompts(update: PromptUpdate):
    """Update text prompts for garment types. Only provided fields are updated."""
    global prompts
    changes = {}
    for field in ["full", "upper", "lower", "layered"]:
        val = getattr(update, field)
        if val is not None:
            prompts[field] = val
            changes[field] = val

    logger.info(f"Prompts updated: {changes}")
    return {"status": "ok", "prompts": prompts, "changes": changes}


@app.post("/config/mask")
async def update_mask_config(update: MaskConfigUpdate):
    """Update mask rendering config. Only provided fields are updated."""
    global mask_config
    changes = {}
    for field in ["green_color", "green_alpha", "blur_radius", "border_width", "threshold", "mask_threshold"]:
        val = getattr(update, field)
        if val is not None:
            if field == "green_color":
                mask_config[field] = tuple(val)
            else:
                mask_config[field] = val
            changes[field] = val

    logger.info(f"Mask config updated: {changes}")
    return {"status": "ok", "mask_config": mask_config, "changes": changes}


@app.get("/config")
async def get_config():
    """Get current prompts and mask config."""
    return {
        "prompts": prompts,
        "mask_config": mask_config,
        "defaults": {
            "prompts": DEFAULT_PROMPTS,
            "mask_config": MASK_CONFIG,
        },
    }


@app.post("/config/reset")
async def reset_config():
    """Reset prompts and mask config to defaults."""
    global prompts, mask_config
    prompts = dict(DEFAULT_PROMPTS)
    mask_config = dict(MASK_CONFIG)
    logger.info("Config reset to defaults")
    return {"status": "ok", "prompts": prompts, "mask_config": mask_config}


# ──────────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
