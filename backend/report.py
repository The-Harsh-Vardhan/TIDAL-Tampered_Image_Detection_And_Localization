"""Forensic PDF report generator for TIDAL inference results."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import Any

from fpdf import FPDF, XPos, YPos

# ── Colour palette ─────────────────────────────────────────────────────────
_RED = (220, 53, 69)
_GREEN = (40, 167, 69)
_ORANGE = (255, 140, 0)
_DARK = (33, 37, 41)
_LIGHT_GREY = (245, 245, 245)
_MID_GREY = (108, 117, 125)
_WHITE = (255, 255, 255)

# Page geometry (A4, margins 15mm each side → printable width = 180mm)
_PW = 180  # printable width in mm


def _safe(text: str) -> str:
    """Strip characters outside Latin-1 so fpdf2's built-in fonts don't crash."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _set_fill(pdf: FPDF, rgb: tuple) -> None:
    pdf.set_fill_color(*rgb)


def _set_text(pdf: FPDF, rgb: tuple) -> None:
    pdf.set_text_color(*rgb)


def _section_title(pdf: FPDF, title: str) -> None:
    pdf.ln(4)
    _set_fill(pdf, _DARK)
    _set_text(pdf, _WHITE)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(_PW, 8, _safe(f"  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    _set_text(pdf, _DARK)
    pdf.ln(2)


def _kv_row(pdf: FPDF, key: str, value: str, shade: bool = False) -> None:
    if shade:
        _set_fill(pdf, _LIGHT_GREY)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(65, 6, _safe(key), border=0, fill=shade)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(_PW - 65, 6, _safe(str(value)), border=0, fill=shade)


def build_report(cache_key: str, cached_result: dict[str, Any]) -> bytes:
    """Build and return a forensic PDF report as raw bytes.

    Args:
        cache_key:      SHA-256 key (used in the title / traceability).
        cached_result:  The full response dict previously stored in the cache
                        (includes mask_base64, heatmap_base64, exif, ela_stats, etc.)
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # ── Header ─────────────────────────────────────────────────────────────
    _set_text(pdf, _DARK)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(_PW, 12, "TIDAL Forensic Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", "", 9)
    _set_text(pdf, _MID_GREY)
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(_PW, 6, f"Generated: {ts}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(_PW, 6, f"Cache key: {cache_key[:32]}...", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    _set_text(pdf, _DARK)
    pdf.ln(4)

    # Horizontal rule
    pdf.set_draw_color(*_MID_GREY)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # ── Verdict banner ─────────────────────────────────────────────────────
    is_tampered: bool = cached_result.get("is_tampered", False)
    needs_review: bool = cached_result.get("needs_review", False)
    confidence: float = cached_result.get("confidence", 0.0)

    if is_tampered:
        verdict_text = "TAMPERED"
        banner_colour = _RED
    elif needs_review:
        verdict_text = "NEEDS REVIEW"
        banner_colour = _ORANGE
    else:
        verdict_text = "AUTHENTIC"
        banner_colour = _GREEN

    _set_fill(pdf, banner_colour)
    _set_text(pdf, _WHITE)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(_PW, 14, verdict_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="C")
    _set_text(pdf, _DARK)
    pdf.ln(4)

    # ── Key metrics ────────────────────────────────────────────────────────
    _section_title(pdf, "Detection Summary")
    metrics = [
        ("Verdict", verdict_text),
        ("Confidence (max pixel prob)", f"{confidence:.4f}"),
        ("Mean probability", f"{cached_result.get('confidence_mean_prob', 0):.4f}"),
        ("Tampered pixel ratio", f"{cached_result.get('tampered_ratio', 0):.4f}"),
        ("Tampered pixel count", str(cached_result.get("tampered_pixel_count", 0))),
        ("Raw tampered pixels (pre-filter)", str(cached_result.get("raw_tampered_pixel_count", 0))),
        ("Area filter triggered", str(cached_result.get("area_filter_triggered", False))),
        ("Needs review", str(needs_review)),
        ("Model version", str(cached_result.get("model_version", "—"))),
        ("Inference time", f"{cached_result.get('inference_time_ms', 0):.1f} ms"),
    ]
    for i, (k, v) in enumerate(metrics):
        _kv_row(pdf, k, v, shade=(i % 2 == 0))
    pdf.ln(2)

    # ── Applied settings ───────────────────────────────────────────────────
    settings: dict = cached_result.get("applied_settings", {})
    if settings:
        _section_title(pdf, "Applied Inference Settings")
        for i, (k, v) in enumerate(settings.items()):
            _kv_row(pdf, k, str(v), shade=(i % 2 == 0))
        pdf.ln(2)

    # ── ELA energy stats ───────────────────────────────────────────────────
    ela_stats: list[dict] = cached_result.get("ela_stats", [])
    if ela_stats:
        _section_title(pdf, "Error Level Analysis (ELA) Energy Stats")
        # Column widths that sum to _PW = 180: 30+50+50+50
        col_w = [30, 50, 50, 50]
        headers = ["Quality", "Mean Energy", "Max Energy", "Scale Factor"]
        _set_fill(pdf, _DARK)
        _set_text(pdf, _WHITE)
        pdf.set_font("Helvetica", "B", 9)
        for i, (h, w) in enumerate(zip(headers, col_w)):
            is_last = i == len(headers) - 1
            pdf.cell(w, 6, h, border=1, fill=True,
                     new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                     new_y=YPos.NEXT if is_last else YPos.TOP)
        _set_text(pdf, _DARK)
        for row_i, stat in enumerate(ela_stats):
            shade = row_i % 2 == 0
            if shade:
                _set_fill(pdf, _LIGHT_GREY)
            pdf.set_font("Helvetica", "", 9)
            row_vals = [
                f"Q={stat.get('quality', '?')}",
                str(stat.get("mean_energy", "—")),
                str(stat.get("max_energy", "—")),
                str(stat.get("scale_factor", "—")),
            ]
            for i, (val, w) in enumerate(zip(row_vals, col_w)):
                is_last = i == len(col_w) - 1
                pdf.cell(w, 6, val, border=1, fill=shade,
                         new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                         new_y=YPos.NEXT if is_last else YPos.TOP)
        pdf.ln(4)

    # ── Threshold sensitivity table ────────────────────────────────────────
    sensitivity: list[dict] = cached_result.get("threshold_sensitivity", [])
    if sensitivity:
        _section_title(pdf, "Threshold Sensitivity Analysis")
        # 36+36+36+36+36 = 180
        col_w2 = [36, 36, 36, 36, 36]
        headers2 = ["Threshold", "Raw Pixels", "Final Pixels", "Ratio", "Area Filtered"]
        _set_fill(pdf, _DARK)
        _set_text(pdf, _WHITE)
        pdf.set_font("Helvetica", "B", 9)
        for i, (h, w) in enumerate(zip(headers2, col_w2)):
            is_last = i == len(headers2) - 1
            pdf.cell(w, 6, h, border=1, fill=True,
                     new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                     new_y=YPos.NEXT if is_last else YPos.TOP)
        _set_text(pdf, _DARK)
        for row_i, row in enumerate(sensitivity):
            shade = row_i % 2 == 0
            if shade:
                _set_fill(pdf, _LIGHT_GREY)
            pdf.set_font("Helvetica", "", 9)
            row_vals2 = [
                str(row.get("threshold", "—")),
                str(row.get("raw_pixels", "—")),
                str(row.get("final_pixels", "—")),
                str(row.get("ratio", "—")),
                str(row.get("area_filtered", "—")),
            ]
            for i, (val, w) in enumerate(zip(row_vals2, col_w2)):
                is_last = i == len(col_w2) - 1
                pdf.cell(w, 6, val, border=1, fill=shade,
                         new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                         new_y=YPos.NEXT if is_last else YPos.TOP)
        pdf.ln(4)

    # ── EXIF / Metadata ────────────────────────────────────────────────────
    exif: dict = cached_result.get("exif", {})
    if exif:
        _section_title(pdf, "EXIF Metadata & Forensic Flags")
        flags: list[str] = exif.get("forensic_flags", [])
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(_PW, 6, "Forensic Observations:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        for flag in flags:
            _set_fill(pdf, _LIGHT_GREY)
            pdf.multi_cell(_PW, 6, _safe(f"  - {flag}"), fill=True)
            pdf.ln(1)

        raw_tags: dict = exif.get("raw", {})
        if raw_tags:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(_PW, 6, "Raw EXIF Tags:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for i, (tag, val) in enumerate(raw_tags.items()):
                _kv_row(pdf, tag, str(val), shade=(i % 2 == 0))
        pdf.ln(2)

    # ── Embedded images ────────────────────────────────────────────────────
    mask_b64: str = cached_result.get("mask_base64", "")
    heatmap_b64: str = cached_result.get("heatmap_base64", "")
    overlay_b64: str = cached_result.get("overlay_base64", "")

    images_to_embed = []
    if mask_b64:
        images_to_embed.append(("Binary Tamper Mask", mask_b64))
    if heatmap_b64:
        images_to_embed.append(("Probability Heatmap (blue=low, red=high)", heatmap_b64))
    if overlay_b64:
        images_to_embed.append(("Overlay (tampered regions highlighted)", overlay_b64))

    if images_to_embed:
        _section_title(pdf, "Visual Evidence")
        for label, b64_data in images_to_embed:
            try:
                img_bytes = base64.b64decode(b64_data)
                img_buf = io.BytesIO(img_bytes)
                from PIL import Image as PILImage
                pil_img = PILImage.open(img_buf)
                w_px, h_px = pil_img.size
                target_w_mm = 85.0
                target_h_mm = target_w_mm * (h_px / w_px)

                img_buf.seek(0)
                pdf.set_font("Helvetica", "I", 9)
                _set_text(pdf, _MID_GREY)
                pdf.cell(_PW, 5, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                _set_text(pdf, _DARK)
                x = pdf.get_x()
                y = pdf.get_y()
                pdf.image(img_buf, x=x, y=y, w=target_w_mm)
                pdf.ln(target_h_mm + 4)
            except Exception:
                pass  # Skip image if decoding fails

    # ── Footer ─────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    _set_text(pdf, _MID_GREY)
    pdf.cell(
        _PW, 8,
        "Generated by TIDAL - Tampered Image Detection & Localization | "
        "github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization",
        align="C",
    )

    return bytes(pdf.output())
