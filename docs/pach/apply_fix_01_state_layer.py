"""apply_fix_01_state_layer.py

state.py に層グループ選択用の変数と定数を追加する。

追加内容:
  定数: LAYER_INPUT_MODES, LAYER_GROUP_DEFS
  変数: layer_input_mode (StringVar), layer_groups (dict[str, BooleanVar])
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
        print(f"[ERROR] patch '{label}' target ambiguous in {path}")
        sys.exit(1)

    result = src.replace(old_n, new_n, 1)
    raw = path.read_bytes()
    eol = "\r\n" if b"\r\n" in raw else "\n"
    path.write_text(result.replace("\n", eol), encoding="utf-8")
    print(f"[OK] patched '{label}' in {path}")


HERE = pathlib.Path(__file__).parent
STATE_PY = HERE / "app" / "state.py"

if not STATE_PY.exists():
    print(f"[ERROR] {STATE_PY} not found")
    sys.exit(1)

# ── ネットワークコメント行を実ファイルから直接取得 ────────────────────────────
# 罫線数がファイルによって異なるため実ファイルの行を直接読んで使用する
_src_lines = STATE_PY.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
_NC = next(
    (l for l in _src_lines if "\u30cd\u30c3\u30c8\u30ef\u30fc\u30af" in l and "\u2500" in l),
    None,
)
if _NC is None:
    print("[ERROR] ネットワーク変数コメント行が state.py に見つかりません")
    sys.exit(1)
print(f"[INFO] network comment line: {repr(_NC)}")

# ── Patch 1: LORA_TARGETS 定数の直後に LAYER_GROUP_DEFS 定数を追加 ───────────
P1_OLD = 'LORA_TARGETS: tuple[str, ...] = ("all", "attention_only", "custom")'

P1_NEW = (
    'LORA_TARGETS: tuple[str, ...] = ("all", "attention_only", "custom")\n'
    '\n'
    '# カスタム選択時のサブモード\n'
    'LAYER_INPUT_MODES: tuple[str, ...] = ("text", "group")\n'
    '\n'
    '# 層グループ定義: (group_key, label, exclude_pattern)\n'
    '# exclude_pattern は lora.py の re.fullmatch に渡す original_name パターン。\n'
    'LAYER_GROUP_DEFS: tuple[tuple[str, str, str], ...] = (\n'
    '    ("first",            "first\uff08\u5165\u529b\u6295\u5f71\uff09",\n'
    '     r"first"),\n'
    '    ("blocks_0_9_attn",  "blocks.0\u301c9  Attention",\n'
    '     r"blocks\\.[0-9]\\.attn\\..*"),\n'
    '    ("blocks_0_9_mlp",   "blocks.0\u301c9  MLP",\n'
    '     r"blocks\\.[0-9]\\.mlp\\..*"),\n'
    '    ("blocks_10_19_attn","blocks.10\u301c19 Attention",\n'
    '     r"blocks\\.(1[0-9])\\.attn\\..*"),\n'
    '    ("blocks_10_19_mlp", "blocks.10\u301c19 MLP",\n'
    '     r"blocks\\.(1[0-9])\\.mlp\\..*"),\n'
    '    ("blocks_20_27_attn","blocks.20\u301c27 Attention",\n'
    '     r"blocks\\.(2[0-7])\\.attn\\..*"),\n'
    '    ("blocks_20_27_mlp", "blocks.20\u301c27 MLP",\n'
    '     r"blocks\\.(2[0-7])\\.mlp\\..*"),\n'
    '    ("txtfusion",        "txtfusion.*\uff08\u30c6\u30ad\u30b9\u30c8\u878d\u5408\uff09",\n'
    '     r"txtfusion\\..*"),\n'
    '    ("tmlp_txtmlp",      "tmlp / txtmlp / tproj\uff08\u6642\u523b\u30fb\u30c6\u30ad\u30b9\u30c8\u6295\u5f71\uff09",\n'
    '     r"(tmlp|txtmlp|tproj)\\..*"),\n'
    '    ("last_linear",      "last.linear\uff08\u51fa\u529b\u6295\u5f71\uff09",\n'
    '     r"last\\.linear"),\n'
    ')\n'
)

_apply(STATE_PY, P1_OLD, P1_NEW, "LAYER_GROUP_DEFS const")


# ── Patch 2: TrainState.__init__ のネットワーク変数ブロックへ追加 ─────────────
P2_OLD = (
    _NC + "\n"
    '        self.network_dim        = tk.IntVar(value=32)\n'
    '        self.network_alpha      = tk.DoubleVar(value=32.0)\n'
    '        self.lora_target        = tk.StringVar(value="all")\n'
    '        self.network_args       = tk.StringVar(value="")\n'
    '        self.network_weights    = tk.StringVar()'
)

_LAYER_COMMENT = "        # \u2500\u2500 \u5c64\u30b0\u30eb\u30fc\u30d7\u9078\u629e\uff08\u30ab\u30b9\u30bf\u30e0\u6642\u30b5\u30d6\u30e2\u30fc\u30c9\uff09" + "\u2500" * 11

P2_NEW = (
    _NC + "\n"
    '        self.network_dim        = tk.IntVar(value=32)\n'
    '        self.network_alpha      = tk.DoubleVar(value=32.0)\n'
    '        self.lora_target        = tk.StringVar(value="all")\n'
    '        self.network_args       = tk.StringVar(value="")\n'
    '        self.network_weights    = tk.StringVar()\n'
    '\n'
    + _LAYER_COMMENT + '\n'
    '        # "text"  : network_args \u30c6\u30ad\u30b9\u30c8\u76f4\u63a5\u5165\u529b\n'
    '        # "group" : \u5c64\u30b0\u30eb\u30fc\u30d7 \u30c1\u30a7\u30c3\u30af\u30dc\u30c3\u30af\u30b9 UI\n'
    '        self.layer_input_mode   = tk.StringVar(value="text")\n'
    '        # \u5404\u30b0\u30eb\u30fc\u30d7\u306e\u30aa\u30f3/\u30aa\u30d5\uff08True=\u5b66\u7fd2\u5bfe\u8c61\u306b\u542b\u3081\u308b\uff09\n'
    '        self.layer_groups: dict[str, tk.BooleanVar] = {\n'
    '            "first":             tk.BooleanVar(value=True),\n'
    '            "blocks_0_9_attn":   tk.BooleanVar(value=True),\n'
    '            "blocks_0_9_mlp":    tk.BooleanVar(value=True),\n'
    '            "blocks_10_19_attn": tk.BooleanVar(value=True),\n'
    '            "blocks_10_19_mlp":  tk.BooleanVar(value=True),\n'
    '            "blocks_20_27_attn": tk.BooleanVar(value=True),\n'
    '            "blocks_20_27_mlp":  tk.BooleanVar(value=True),\n'
    '            "txtfusion":         tk.BooleanVar(value=True),\n'
    '            "tmlp_txtmlp":       tk.BooleanVar(value=True),\n'
    '            "last_linear":       tk.BooleanVar(value=True),\n'
    '        }'
)

_apply(STATE_PY, P2_OLD, P2_NEW, "layer_groups vars")

print("\n[apply_fix_01] state.py のパッチ完了。")
