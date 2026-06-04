"""EXIF metadata extraction and forensic flag analysis for TIDAL."""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Software strings that strongly indicate post-processing / editing
_EDITING_SOFTWARE_KEYWORDS = (
    "photoshop",
    "lightroom",
    "gimp",
    "affinity",
    "pixelmator",
    "paint.net",
    "darktable",
    "capture one",
    "luminar",
    "snapseed",
    "facetune",
    "meitu",
    "picsart",
)

# EXIF tag IDs we care about (subset of the full ~300-tag spec)
_TAGS_OF_INTEREST = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "ExifImageWidth", "ExifImageHeight",
    "PixelXDimension", "PixelYDimension", "GPSInfo",
    "Orientation", "ColorSpace", "Flash", "FocalLength",
    "ISOSpeedRatings", "ExposureTime", "FNumber", "Artist",
    "Copyright", "ImageDescription", "UserComment", "XResolution",
    "YResolution",
}


def _decode_exif_value(value: Any) -> Any:
    """Convert IFDRational, bytes, and other EXIF types to plain Python objects."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return value.hex()
    if hasattr(value, "numerator"):  # IFDRational
        try:
            return float(value)
        except Exception:
            return str(value)
    if isinstance(value, tuple):
        return [_decode_exif_value(v) for v in value]
    return value


def extract_exif(raw_bytes: bytes) -> dict:
    """Extract EXIF metadata from raw image bytes.

    Returns a dict with:
        ``raw``            – dict of tag_name → decoded value (only tags of interest)
        ``has_exif``       – bool
        ``has_gps``        – bool
        ``forensic_flags`` – list[str] of human-readable forensic observations
    """
    result: dict = {
        "raw": {},
        "has_exif": False,
        "has_gps": False,
        "forensic_flags": [],
    }

    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        logger.debug("EXIF: could not open image — %s", exc)
        return result

    # ── Extract raw EXIF dict ──────────────────────────────────────────────
    exif_data: dict | None = None
    try:
        exif_data = img._getexif()  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Format doesn't support EXIF (PNG, WebP without XMP, etc.)
    except Exception as exc:
        logger.debug("EXIF: _getexif() failed — %s", exc)

    if not exif_data:
        result["forensic_flags"].append(
            "No EXIF metadata found — image may have been stripped or re-saved without metadata."
        )
        return result

    result["has_exif"] = True

    # Decode tags of interest
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        if tag_name == "GPSInfo":
            result["has_gps"] = True
            result["raw"]["GPSInfo"] = "(present)"
            continue
        if tag_name in _TAGS_OF_INTEREST:
            result["raw"][tag_name] = _decode_exif_value(value)

    # ── Forensic flag analysis ─────────────────────────────────────────────
    flags: list[str] = []
    raw = result["raw"]

    # 1. Editing software detected
    software = str(raw.get("Software", "")).lower()
    if software:
        for kw in _EDITING_SOFTWARE_KEYWORDS:
            if kw in software:
                flags.append(
                    f"Editing software detected in EXIF: '{raw['Software']}'. "
                    "Image may have been processed or composited."
                )
                break

    # 2. No GPS on an image that claims to be from a phone camera
    make = str(raw.get("Make", "")).lower()
    model_str = str(raw.get("Model", "")).lower()
    is_likely_phone = any(
        kw in make or kw in model_str
        for kw in ("apple", "samsung", "huawei", "xiaomi", "google", "oneplus", "oppo", "vivo")
    )
    if is_likely_phone and not result["has_gps"]:
        flags.append(
            "Camera identified as a smartphone but GPS data is absent — "
            "GPS may have been stripped (potential metadata removal)."
        )

    # 3. DateTime vs. DateTimeOriginal mismatch
    dt = raw.get("DateTime")
    dt_orig = raw.get("DateTimeOriginal")
    if dt and dt_orig and dt != dt_orig:
        flags.append(
            f"Timestamp mismatch: DateTime='{dt}' differs from DateTimeOriginal='{dt_orig}'. "
            "File may have been re-saved after capture."
        )

    # 4. Software present but no Make/Model (no camera info)
    if software and not raw.get("Make") and not raw.get("Model"):
        flags.append(
            "Software tag present but camera Make/Model is absent — "
            "likely generated or heavily edited, not a direct camera capture."
        )

    # 5. Declared pixel dimensions vs. actual
    declared_w = raw.get("ExifImageWidth") or raw.get("PixelXDimension")
    declared_h = raw.get("ExifImageHeight") or raw.get("PixelYDimension")
    if declared_w and declared_h:
        try:
            if int(declared_w) != img.width or int(declared_h) != img.height:
                flags.append(
                    f"Declared EXIF dimensions ({int(declared_w)}×{int(declared_h)}) "
                    f"differ from actual image size ({img.width}×{img.height}). "
                    "Image may have been cropped or resized."
                )
        except (TypeError, ValueError):
            pass

    if not flags:
        flags.append("No forensic anomalies detected in EXIF metadata.")

    result["forensic_flags"] = flags
    return result
