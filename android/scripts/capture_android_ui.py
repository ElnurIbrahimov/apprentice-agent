#!/usr/bin/env python
"""Capture deterministic Aura debug-catalog screenshots with foreground proof."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

PACKAGE = "com.aura.debug"
ACTIVITY = "com.aura.debug.UiCatalogActivity"
SURFACES = {
    "startup", "onboarding", "home", "chat", "model-picker", "settings",
    "memory", "history", "tasks", "reminders", "hands", "tools",
    "proactive", "graph", "profile", "identity", "diagnostics", "voice",
    "quick-ask", "widget-config",
}
STATES = {"content", "loading", "empty", "error", "no-provider", "selected", "partial-error"}
THEMES = {"light", "dark"}


@dataclass(frozen=True)
class Shot:
    surface: str
    state: str
    theme: str

    @property
    def filename(self) -> str:
        return f"{self.surface}__{self.state}__{self.theme}.png"


CURATED_MATRIX = (
    Shot("home", "content", "light"),
    Shot("home", "content", "dark"),
    Shot("chat", "content", "dark"),
    Shot("model-picker", "no-provider", "light"),
    Shot("model-picker", "partial-error", "dark"),
    Shot("settings", "content", "light"),
    Shot("memory", "empty", "dark"),
    Shot("tasks", "loading", "light"),
)


class Adb:
    def __init__(self, executable: str, serial: str | None) -> None:
        self.prefix = [executable]
        if serial:
            self.prefix += ["-s", serial]

    def run(self, *args: str, binary: bool = False) -> bytes | str:
        result = subprocess.run([*self.prefix, *args], check=True, capture_output=True)
        return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")


def validate(shot: Shot) -> None:
    if shot.surface not in SURFACES:
        raise ValueError(f"Unknown surface: {shot.surface}")
    if shot.state not in STATES:
        raise ValueError(f"Unknown state: {shot.state}")
    if shot.theme not in THEMES:
        raise ValueError(f"Unknown theme: {shot.theme}")


def assert_foreground(adb: Adb) -> None:
    activities = str(adb.run("shell", "dumpsys", "activity", "activities"))
    resumed = [
        line.strip()
        for line in activities.splitlines()
        if "mResumedActivity" in line or "topResumedActivity" in line
    ]
    full = f"{PACKAGE}/{ACTIVITY}"
    short = f"{PACKAGE}/.UiCatalogActivity"
    if not any(full in line or short in line for line in resumed):
        raise RuntimeError(f"UI catalog is not resumed; resumed records: {resumed}")

    raw_dump = ""
    for _ in range(5):
        time.sleep(0.4)
        raw_dump = str(adb.run("exec-out", "uiautomator", "dump", "/dev/tty"))
        if "<?xml" in raw_dump and "</hierarchy>" in raw_dump:
            break
    start = raw_dump.find("<?xml")
    end = raw_dump.find("</hierarchy>")
    if start < 0 or end < 0:
        windows = str(adb.run("shell", "dumpsys", "window"))
        focused = [
            line.strip()
            for line in windows.splitlines()
            if "mCurrentFocus" in line or "mFocusedApp" in line
        ]
        if not any(PACKAGE in line for line in focused):
            raise RuntimeError(
                f"No Aura UI root or focused window; dump={raw_dump[:120]!r}, focus={focused}",
            )
        return
    xml_text = raw_dump[start : end + len("</hierarchy>")]
    root = ET.fromstring(xml_text)
    packages = {node.attrib.get("package") for node in root.iter("node")}
    if packages != {PACKAGE}:
        raise RuntimeError(f"UIAutomator root is not exclusively Aura: {sorted(packages)}")


def capture(adb: Adb, shot: Shot, output: pathlib.Path) -> None:
    validate(shot)
    adb.run("shell", "am", "force-stop", PACKAGE)
    start = str(adb.run(
        "shell", "am", "start", "-W",
        "-n", f"{PACKAGE}/{ACTIVITY}",
        "--es", "surface", shot.surface,
        "--es", "state", shot.state,
        "--es", "theme", shot.theme,
    ))
    if "UiCatalogActivity" not in start:
        raise RuntimeError(f"Activity launch did not name the catalog:\n{start}")
    time.sleep(0.35)
    assert_foreground(adb)
    # Compose modal sheets can become focused one frame before their enter transition paints.
    time.sleep(0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    png = adb.run("exec-out", "screencap", "-p", binary=True)
    assert isinstance(png, bytes)
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 5_000:
        raise RuntimeError(f"Invalid screenshot payload ({len(png)} bytes)")
    output.write_bytes(png)
    print(output.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial")
    parser.add_argument("--apk", required=True)
    parser.add_argument("--surface", default="home")
    parser.add_argument("--state", default="content")
    parser.add_argument("--theme", choices=sorted(THEMES), default="dark")
    parser.add_argument("--out")
    parser.add_argument("--out-dir", default=".hermes/screenshots/catalog")
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()

    adb = Adb(args.adb, args.serial)
    adb.run("install", "-r", args.apk)

    if args.matrix:
        out_dir = pathlib.Path(args.out_dir)
        for shot in CURATED_MATRIX:
            capture(adb, shot, out_dir / shot.filename)
    else:
        shot = Shot(args.surface, args.state, args.theme)
        output = pathlib.Path(args.out) if args.out else pathlib.Path(args.out_dir) / shot.filename
        capture(adb, shot, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
