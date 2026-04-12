"""Shared pytest configuration for warning hygiene."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 \(.*\) or chardet \(.*\)/charset_normalizer \(.*\) doesn't match a supported version",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)
