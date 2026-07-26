#!/usr/bin/env python3
"""Smoke test that every wrapper module has an explicit bundle owner."""

from __future__ import annotations

from pathlib import Path

from dataset_clip_wrapper.module_bundles import bundle_by_module, python_modules


def test_module_bundle_coverage() -> None:
    package_dir = Path(__file__).resolve().parents[1]
    modules = python_modules(package_dir)
    expected = set(bundle_by_module())
    ignored = {"__init__", "module_bundles"}
    missing = sorted(modules - expected - ignored)
    stale = sorted(expected - modules)
    assert not missing, f"Unclassified dataset_clip_wrapper modules: {missing}"
    assert not stale, f"Bundle registry mentions missing modules: {stale}"


if __name__ == "__main__":
    test_module_bundle_coverage()
    print("module bundle smoke test passed")
