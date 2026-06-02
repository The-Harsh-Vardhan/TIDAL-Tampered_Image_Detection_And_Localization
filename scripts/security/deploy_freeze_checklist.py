#!/usr/bin/env python3
"""Run deployment freeze readiness checks for incident response."""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate readiness before unfreezing deploys.")
    parser.add_argument("--api-base", default="", help="Base API URL for endpoint checks.")
    parser.add_argument(
        "--run-local-checks",
        action="store_true",
        help="Run pytest, pip-audit, and ruff checks.",
    )
    parser.add_argument(
        "--confirm-freeze",
        action="store_true",
        help="Confirm production deploy freeze is active in provider dashboard.",
    )
    parser.add_argument(
        "--confirm-rotation",
        action="store_true",
        help="Confirm token/key rotation is completed.",
    )
    parser.add_argument(
        "--confirm-env-audit",
        action="store_true",
        help="Confirm environment variable audit/remediation is completed.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any check fails.",
    )
    return parser.parse_args()


def run_command(name: str, command: list[str]) -> CheckResult:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return CheckResult(name=name, ok=True, detail="pass")

    output = (result.stderr or result.stdout).strip().splitlines()
    detail = output[-1] if output else f"exit_code={result.returncode}"
    return CheckResult(name=name, ok=False, detail=detail)


def check_manual(name: str, confirmed: bool) -> CheckResult:
    return CheckResult(name=name, ok=confirmed, detail="confirmed" if confirmed else "not confirmed")


def check_endpoint(base: str, path: str) -> CheckResult:
    url = f"{base.rstrip('/')}{path}"
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            status = response.getcode()
    except urllib.error.URLError as exc:
        return CheckResult(name=f"endpoint {path}", ok=False, detail=str(exc.reason))

    ok = status == 200
    return CheckResult(name=f"endpoint {path}", ok=ok, detail=f"status={status}")


def print_results(results: list[CheckResult]) -> None:
    print("Deploy Freeze Checklist")
    for result in results:
        marker = "[x]" if result.ok else "[ ]"
        print(f"{marker} {result.name}: {result.detail}")


def main() -> int:
    args = parse_args()
    results: list[CheckResult] = [
        check_manual("Production deploy freeze active", args.confirm_freeze),
        check_manual("Credential rotation complete", args.confirm_rotation),
        check_manual("Environment audit complete", args.confirm_env_audit),
    ]

    if args.run_local_checks:
        results.extend(
            [
                run_command("pytest tests/ -v", ["pytest", "tests/", "-v"]),
                run_command("pip-audit --strict", ["pip-audit", "--strict"]),
                run_command("ruff check .", ["ruff", "check", "."]),
            ]
        )

    if args.api_base:
        results.extend(
            [
                check_endpoint(args.api_base, "/health"),
                check_endpoint(args.api_base, "/ready"),
                check_endpoint(args.api_base, "/version"),
                check_endpoint(args.api_base, "/metrics"),
            ]
        )

    print_results(results)
    has_failures = any(not item.ok for item in results)

    if has_failures:
        print("One or more checks failed.")
    else:
        print("All checks passed.")

    return 1 if args.strict and has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
