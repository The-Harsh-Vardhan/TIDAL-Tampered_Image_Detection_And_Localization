"""Security utilities - validation, rate limiting."""
from __future__ import annotations
import logging, time, os
from collections import defaultdict
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES = {"image/jpeg","image/png","image/webp","image/jpg"}
ALLOWED_EXTENSIONS = {".jpg",".jpeg",".png",".webp"}
MAX_FILE_SIZE_BYTES = 20*1024*1024
MAX_IMAGE_PIXELS = 16_000_000
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE","30"))

def validate_file_type(content_type, filename):
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported type: {content_type}")
    if filename:
        ext = "."+filename.rsplit(".",1)[-1].lower() if "." in filename else ""
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported extension: {ext}")

def validate_file_size(size):
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, f"File too large: {size/1024/1024:.1f}MB. Max: 20MB")

def validate_image_dimensions(width, height):
    if width*height > MAX_IMAGE_PIXELS:
        raise HTTPException(413, f"Image too large: {width}x{height}")

class RateLimiter:
    def __init__(self, max_requests=RATE_LIMIT_PER_MINUTE, window_seconds=60):
        self.max_requests=max_requests; self.window_seconds=window_seconds
        self._requests = defaultdict(list)
    def check(self, request: Request):
        ip = request.headers.get("X-Forwarded-For","").split(",")[0].strip() or (request.client.host if request.client else "unknown")
        now = time.time(); cutoff = now-self.window_seconds
        self._requests[ip] = [t for t in self._requests[ip] if t>cutoff]
        self._requests[ip].append(now)
        if len(self._requests[ip]) > self.max_requests:
            raise HTTPException(429, f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s")
