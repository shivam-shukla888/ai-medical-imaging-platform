"""
Tests for React frontend production build artifacts.
"""
from pathlib import Path
import pytest


def test_frontend_production_build_artifacts():
    """Verify that frontend/dist contains index.html and compiled JS/CSS bundles."""
    dist_dir = Path("frontend/dist")
    assert dist_dir.is_dir(), "frontend/dist directory does not exist. Run 'npm run build' in frontend/."

    index_html = dist_dir / "index.html"
    assert index_html.is_file(), "frontend/dist/index.html missing"

    content = index_html.read_text(encoding="utf-8")
    assert '<div id="root"></div>' in content
    assert "AI Chest X-Ray Research Platform" in content

    assets_dir = dist_dir / "assets"
    assert assets_dir.is_dir(), "frontend/dist/assets missing"
    js_files = list(assets_dir.glob("*.js"))
    css_files = list(assets_dir.glob("*.css"))
    assert len(js_files) >= 1, "Compiled JS bundle missing"
    assert len(css_files) >= 1, "Compiled CSS bundle missing"
