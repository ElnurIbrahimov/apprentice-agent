"""
AURA Extension Build Script
Usage:
  python build.py          → builds Chrome  (dist-chrome/)
  python build.py firefox  → builds Firefox (dist-firefox/)
"""

import shutil
import sys

target = sys.argv[1] if len(sys.argv) > 1 else 'chrome'
dst = f'dist-{target}'

shutil.copytree('extension', dst, dirs_exist_ok=True)

if target == 'firefox':
    shutil.copy('extension/manifest.firefox.json', f'{dst}/manifest.json')
    print(f'Built firefox -> {dst}/ (Firefox manifest applied)')
else:
    print(f'Built chrome -> {dst}/')
