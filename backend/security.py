"""
backend/security.py
====================
Security utilities for the TIDAL API.

Input validation, file-type whitelisting, size guards,
and rate limiting helpers.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_IMAGE_PIXELS = 16_000_000  # 16 megapixels (4000×4000)

# Rate limiting
RATE_LIMIT_PER_MINUTE = int(__import__("os").environ.get("RATE_LIMIT_PER_MINUTE", "30"))
RATE_WINDOW_SECONDS = 60


def validate_file_type(content_type: str | None, filename: str | None) -> None:
    """Validate that the uploaded file is an allowed image type.

    Raises HTTPException 415 if invalid.
    """
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file extension: {ext}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )


def validate_file_size(size: int) -> None:
    """Validate file size is within limits.

    Raises HTTPException 413 if too large.
    """
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size / 1024 / 1024:.1f} MB. "
            f"Maximum: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB",
        )


def validate_image_dimensions(width: int, height: int) -> None:
    """Validate image pixel count is within limits.

    Raises HTTPException 413 if too many pixels.
    """
    total_pixels = width * height
    if total_pixels > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {width}×{height} = {total_pixels:,} pixels. "
            f"Maximum: {MAX_IMAGE_PIXELS:,} pixels",
        )


class RateLimiter:
    """Simple in-memory sliding window rate limiter per client IP."""

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_PER_MINUTE,
        window_seconds: int = RATE_WINDOW_SECONDS,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        """Check rate limit for a request.

        Raises HTTPException 429 if limit exceeded.
        """
        client_ip = self._get_client_ip(request)
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean old entries and add current
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]
        self._requests[client_ip].append(now)

        if len(self._requests[client_ip]) > self.max_requests:
            logger.warning("Rate limit exceeded for %s", client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.max_requests} "
                f"requests per {self.window_seconds} seconds.",
            )

    def cleanup(self) -> None:
        """Remove stale entries to prevent memory leak."""
        now = time.time()
        cutoff = now - self.window_seconds
        stale_keys = [
            ip
            for ip, timestamps in self._requests.items()
            if all(t <= cutoff for t in timestamps)
        ]
        for key in stale_keys:
            del self._requests[key]
