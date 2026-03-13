"""
AURA Extension Build Script
Usage:
  python build.py          → builds Chrome  (dist-chrome/)
  python build.py firefox  → builds Firefox (dist-firefox/)
"""

import shutil
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Build the React frontend
ext_src = os.path.join(SCRIPT_DIR, 'extension-src')
print('Building React frontend…')
subprocess.run('npm run build', cwd=ext_src, check=True, shell=True)
print('React build complete.')

target = sys.argv[1] if len(sys.argv) > 1 else 'chrome'
dst = os.path.join(SCRIPT_DIR, f'dist-{target}')
ext_dir = os.path.join(SCRIPT_DIR, 'extension')

# Step 2: Copy extension/ to dist
shutil.copytree(ext_dir, dst, dirs_exist_ok=True)

if target == 'firefox':
    shutil.copy(os.path.join(ext_dir, 'manifest.firefox.json'), os.path.join(dst, 'manifest.json'))
    print(f'Built firefox -> {dst}/ (Firefox manifest applied)')
else:
    print(f'Built chrome -> {dst}/')
