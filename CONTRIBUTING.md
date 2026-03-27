# Contributing to TIDAL

Thanks for your interest! Here's how to contribute.

## Quick Start

```bash
git clone https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization.git
cd TIDAL-Tampered_Image_Detection_And_Localization
git checkout -b your-feature-branch
pip install -e ".[dev]"
```

## Workflow

1. **Fork** the repository
2. **Branch** from `production` — `git checkout -b feat/your-feature`
3. **Code** — follow ruff formatting (`ruff format .`)
4. **Test** — `pytest tests/ -v`
5. **Lint** — `ruff check .`
6. **PR** — open a pull request against `production`

## Code Style

- Python: enforced by [Ruff](https://docs.astral.sh/ruff/) (config in `pyproject.toml`)
- Frontend: vanilla HTML/CSS/JS, no frameworks
- Commits: use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.)

## Reporting Issues

Open a GitHub Issue with:
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, GPU)

For security issues, see [SECURITY.md](SECURITY.md).
