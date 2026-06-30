"""apply_fix_04_preset_layer.py

preset_manager.py の collect_preset / apply_preset に
層グループ選択変数を追加する。

追加されるプリセットキー:
  "layer_input_mode" : str  ("text" | "group")
  "layer_groups"     : dict[str, bool]
"""
from __future__ import annotations

import pathlib
import sys


def _adapt(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _apply(path: pathlib.Path, old: str, new: str, label: str = "") -> None:
    src = _adapt(path.read_text(encoding="utf-8"))
    old_n = _adapt(old)
    new_n = _adapt(new)

    if old_n not in src:
        print(f"[ERROR] patch '{label}' target not found in {path}")
        sys.exit(1)
    if src.count(old_n) > 1:
        print(f"[ERROR] patch '{label}' target ambiguous ({src.count(old_n)}) in {path}")
        sys.exit(1)

    result = src.replace(old_n, new_n, 1)
    raw = path.read_bytes()
    eol = "\r\n" if b"\r\n" in raw else "\n"
    path.write_text(result.replace("\n", eol), encoding="utf-8")
    print(f"[OK] patched '{label}' in {path}")


HERE = pathlib.Path(__file__).parent
PRESET_PY = HERE / "app" / "preset_manager.py"

if not PRESET_PY.exists():
    print(f"[ERROR] {PRESET_PY} not found")
    sys.exit(1)


# ── Patch 1: collect_preset に layer_input_mode / layer_groups を追加 ────────
P1_OLD = (
    '        "network_args":       s.network_args.get(),\n'
    '        "network_weights":    s.network_weights.get(),'
)
P1_NEW = (
    '        "network_args":       s.network_args.get(),\n'
    '        "network_weights":    s.network_weights.get(),\n'
    '        # 層グループ選択\n'
    '        "layer_input_mode":   s.layer_input_mode.get(),\n'
    '        "layer_groups":       {\n'
    '            key: var.get() for key, var in s.layer_groups.items()\n'
    '        },'
)
_apply(PRESET_PY, P1_OLD, P1_NEW, "collect layer_groups")


# ── Patch 2: apply_preset に layer_input_mode / layer_groups 復元を追加 ──────
P2_OLD = (
    '    _set(s.network_args,       "network_args",       "")\n'
    '    _set(s.network_weights,    "network_weights",    "")'
)
P2_NEW = (
    '    _set(s.network_args,       "network_args",       "")\n'
    '    _set(s.network_weights,    "network_weights",    "")\n'
    '    _set(s.layer_input_mode,   "layer_input_mode",   "text")\n'
    '    if "layer_groups" in data and isinstance(data["layer_groups"], dict):\n'
    '        for key, val in data["layer_groups"].items():\n'
    '            if key in s.layer_groups and isinstance(val, bool):\n'
    '                s.layer_groups[key].set(val)'
)
_apply(PRESET_PY, P2_OLD, P2_NEW, "apply layer_groups")

print("\n[apply_fix_04] preset_manager.py のパッチ完了。")
