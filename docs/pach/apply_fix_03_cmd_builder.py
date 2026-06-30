"""apply_fix_03_cmd_builder.py

cmd_builder.py に以下を追加する（未適用のオリジナルファイルに適用）:
  1. import 行に LAYER_GROUP_DEFS を追加
  2. LoRA ターゲットブロックを拡張（層グループ選択対応）
  3. _build_network_args_from_groups() を追加

CRLF 安全設計:
  - ファイル読み込みは常に LF に正規化して比較
  - 書き戻し時は元ファイルの改行コードを保持
  - replace() は正規化済み文字列のみに適用し、最後に改行コードを戻す
"""
from __future__ import annotations

import pathlib
import sys


def _read_lf(path: pathlib.Path) -> str:
    """ファイルを UTF-8 で読み、改行を LF に正規化して返す。"""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _detect_eol(path: pathlib.Path) -> str:
    """元ファイルの改行コードを検出して返す。"""
    raw = path.read_bytes()
    return "\r\n" if b"\r\n" in raw else "\n"


def _apply(path: pathlib.Path, old: str, new: str, label: str = "") -> None:
    """LF 正規化済み old を new に置換し、元の改行コードで書き戻す。"""
    src  = _read_lf(path)
    # old/new も念のため正規化
    old_n = old.replace("\r\n", "\n").replace("\r", "\n")
    new_n = new.replace("\r\n", "\n").replace("\r", "\n")

    cnt = src.count(old_n)
    if cnt == 0:
        print(f"[ERROR] patch '{label}' target not found in {path}")
        sys.exit(1)
    if cnt > 1:
        print(f"[ERROR] patch '{label}' target ambiguous ({cnt}) in {path}")
        sys.exit(1)

    result = src.replace(old_n, new_n, 1)
    eol = _detect_eol(path)
    # LF → 元の改行コードに戻して書き込む
    path.write_text(result.replace("\n", eol), encoding="utf-8")
    print(f"[OK] patched '{label}' in {path}")


HERE   = pathlib.Path(__file__).parent
CMD_PY = HERE / "app" / "cmd_builder.py"

if not CMD_PY.exists():
    print(f"[ERROR] {CMD_PY} not found")
    sys.exit(1)

# 適用前にファイルの状態を確認（P1/P2 が適用済みか否か）
_src_check = _read_lf(CMD_PY)
_already_p1 = "LAYER_GROUP_DEFS" in _src_check
_already_p2 = "layer_input_mode" in _src_check
_already_p3 = "_build_network_args_from_groups" in _src_check

if _already_p1 or _already_p2 or _already_p3:
    print("[ERROR] cmd_builder.py に既にパッチが部分適用されています。")
    print(f"        LAYER_GROUP_DEFS in file : {_already_p1}")
    print(f"        layer_input_mode in file  : {_already_p2}")
    print(f"        _build_network_args in file: {_already_p3}")
    print()
    print("  オリジナルの cmd_builder.py（バックアップ）から適用し直してください。")
    print("  バックアップが存在しない場合は apply_fix_03c_restore_and_patch.py を実行してください。")
    sys.exit(1)

# ── ヘルパーコメント行を動的取得 ─────────────────────────────────────────────
_lines = _src_check.splitlines()
_HC = next((l for l in _lines if "内部ヘルパー" in l and "─" in l), None)
_LTC = next((l for l in _lines if "LoRA" in l and "ターゲット" in l and "─" in l), None)

if _HC is None or _LTC is None:
    print("[ERROR] コメント行が見つかりません")
    sys.exit(1)

# ── Patch 1: import ──────────────────────────────────────────────────────────
_apply(CMD_PY,
    "from .state import TrainState, DATASET_MODE_TOML, SAMPLE_FIXED_SEED",
    "from .state import TrainState, DATASET_MODE_TOML, SAMPLE_FIXED_SEED, LAYER_GROUP_DEFS",
    "import LAYER_GROUP_DEFS")

# ── Patch 2: LoRA ターゲットブロック ─────────────────────────────────────────
_apply(CMD_PY,
    _LTC + "\n"
    "    target = s.lora_target.get()\n"
    "    if target == \"attention_only\":\n"
    "        cmd += [\"--network_args\", _LORA_EXCLUDE_ATTENTION_ONLY]\n"
    "    elif target == \"custom\" and s.network_args.get().strip():\n"
    "        cmd += [\"--network_args\", s.network_args.get().strip()]",
    _LTC + "\n"
    "    target = s.lora_target.get()\n"
    "    if target == \"attention_only\":\n"
    "        cmd += [\"--network_args\", _LORA_EXCLUDE_ATTENTION_ONLY]\n"
    "    elif target == \"custom\":\n"
    "        if s.layer_input_mode.get() == \"group\":\n"
    "            # 層グループ選択モード: OFF グループから exclude_patterns を生成\n"
    "            args_str = _build_network_args_from_groups(s)\n"
    "            if args_str:\n"
    "                cmd += [\"--network_args\", args_str]\n"
    "            # args_str 空 = 全 ON → --network_args なし（全層対象）\n"
    "        else:\n"
    "            # テキスト直接入力モード: network_args をそのまま使用\n"
    "            if s.network_args.get().strip():\n"
    "                cmd += [\"--network_args\", s.network_args.get().strip()]",
    "lora target block")

# ── Patch 3: _build_network_args_from_groups を追加 ──────────────────────────
# P2 後のファイルから HC と _append_optional_str の間の空行を動的検出
_src_after_p2 = _read_lf(CMD_PY)
_hc_idx      = _src_after_p2.index(_HC)
_append_mark = "def _append_optional_str(cmd: list[str], flag: str, value: str) -> None:"
_append_idx  = _src_after_p2.index(_append_mark, _hc_idx)
_gap         = _src_after_p2[_hc_idx + len(_HC):_append_idx]
print(f"[INFO] HC と _append_optional_str の間の空白: {repr(_gap)}")

HELPER_FUNC = (
    "def _build_network_args_from_groups(s: TrainState) -> str:\n"
    '    """層グループ選択の状態から network_args 文字列を生成して返す。\n'
    "\n"
    "    Returns:\n"
    "        exclude_patterns=[...] 形式の文字列。\n"
    "        全グループ ON の場合は空文字（--network_args 不要）。\n"
    '    """\n'
    "    off_keys = [key for key, var in s.layer_groups.items() if not var.get()]\n"
    "    if not off_keys:\n"
    '        return ""\n'
    "\n"
    "    pat_map = {key: pat for key, _label, pat in LAYER_GROUP_DEFS}\n"
    "    patterns = [pat_map[k] for k in off_keys if k in pat_map]\n"
    "    if not patterns:\n"
    '        return ""\n'
    "\n"
    "    quoted = \", \".join(\"'\" + p + \"'\" for p in patterns)\n"
    '    return "exclude_patterns=[" + quoted + "]"\n'
    "\n"
    "\n"
)

_apply(CMD_PY,
    _HC + _gap + _append_mark,
    _HC + "\n\n" + HELPER_FUNC + _append_mark,
    "_build_network_args_from_groups")

print("\n[apply_fix_03] cmd_builder.py のパッチ完了。")
