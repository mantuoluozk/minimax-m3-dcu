"""Materialize the restored D runtime from P source and the recorded overlay."""
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'sglang-das/python'
overlay = root / 'deploy/decode-overlay'
target = root / 'runtime/decode'
if target.exists():
    raise SystemExit(f'{target} already exists; build from a clean checkout')
target.parent.mkdir(exist_ok=True)
shutil.copytree(source / 'sglang', target / 'sglang')
shutil.copytree(overlay, target, dirs_exist_ok=True)
