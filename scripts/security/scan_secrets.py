#!/usr/bin/env python3
"""Scan repository text files for likely hardcoded secrets."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "models",
    "experiments",
    "_archive",
    ".next",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".env",
    ".md",
    ".txt",
    ".cfg",
    ".conf",
    ".sh",
    ".ps1",
}

PLACEHOLDER_HINTS = {
    "your_",
    "your-",
    "example",
    "sample",
    "dummy",
    "changeme",
    "replace_me",
    "placeholder",
    "test",
    "localhost",
    "127.0.0.1",
}


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    file: Path
    line: int
    rule: str
    value: str


RULES = (
    Rule("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b")),
    Rule("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{20,255}\b")),
    Rule("vercel_token", re.compile(r"\bvercel_[A-Za-z0-9]{20,255}\b", re.IGNORECASE)),
    Rule(
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    Rule(
        "assignment_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|"
            r"secret|password|client[_-]?secret|private[_-]?key)\b"
            r"\s*[:=]\s*[\"']?([^\s\"']{12,})"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan files for likely hardcoded secrets.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to scan (default: current directory).",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1_000_000,
        help="Skip files larger than this many bytes (default: 1000000).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print scan progress.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional list of relative files to scan instead of recursive scanning.",
    )
    return parser.parse_args()


def should_scan_file(path: Path, max_file_size: int) -> bool:
    suffix = path.suffix.lower()
    if suffix not in TEXT_EXTENSIONS and path.name not in {".env", ".env.local"}:
        return False

    if path.stat().st_size > max_file_size:
        return False

    return True


def iter_files(root: Path, max_file_size: int) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        try:
            if not path.is_file():
                continue
        except OSError:
            continue

        if should_scan_file(path, max_file_size):
            yield path


def iter_selected_files(root: Path, files: list[str], max_file_size: int) -> Iterable[Path]:
    for raw in files:
        path = (root / raw).resolve()
        if not path.exists() or not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if should_scan_file(path, max_file_size):
            yield path


def extract_value(rule: Rule, match: re.Match[str]) -> str:
    if rule.name == "assignment_secret" and match.lastindex:
        return match.group(1)
    return match.group(0)


def looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    if any(hint in lowered for hint in PLACEHOLDER_HINTS):
        return True
    if value.startswith("${") or value.startswith("{{"):
        return True
    if set(value) in ({"*"}, {"x"}, {"X"}):
        return True
    return False


def mask_value(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings

    for line_no, line in enumerate(content.splitlines(), start=1):
        for rule in RULES:
            for match in rule.pattern.finditer(line):
                value = extract_value(rule, match)
                if looks_like_placeholder(value):
                    continue
                findings.append(Finding(file=path, line=line_no, rule=rule.name, value=value))

    return findings


def print_report(root: Path, findings: list[Finding]) -> None:
    if not findings:
        print("No likely secrets found.")
        return

    print("Potential secrets detected:")
    for finding in findings:
        rel = finding.file.relative_to(root)
        print(
            f"- {rel}:{finding.line} [{finding.rule}] value={mask_value(finding.value)}"
        )


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    findings: list[Finding] = []
    files_to_scan: Iterable[Path]
    if args.files:
        files_to_scan = iter_selected_files(root, args.files, args.max_file_size)
    else:
        files_to_scan = iter_files(root, args.max_file_size)

    for file_path in files_to_scan:
        if args.verbose:
            print(f"Scanning {file_path.relative_to(root)}")
        findings.extend(scan_file(file_path))

    print_report(root, findings)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
