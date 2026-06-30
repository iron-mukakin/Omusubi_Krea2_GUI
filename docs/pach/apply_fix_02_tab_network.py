"""apply_fix_02_tab_network.py

tab_network.py を全面置き換えする。

変更点:
  - RadioButton 3択 → Combobox プルダウン（排他制御）
  - カスタム選択時にサブモードラジオ「テキスト直接入力 / 層グループ選択」を表示
  - 層グループ選択 UI: チェックボックス群 + 全ON/全OFF ボタン + プレビュー
  - プリセット保存対応（state.layer_groups / layer_input_mode を経由）
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).parent
TARGET = HERE / "app" / "tab_network.py"

if not TARGET.exists():
    print(f"[ERROR] {TARGET} not found")
    sys.exit(1)

LINES = [
    '"""app/tab_network.py — [ネットワーク] タブ UI ビルダー。"""',
    'from __future__ import annotations',
    '',
    'import tkinter as tk',
    'from tkinter import ttk, messagebox',
    '',
    'from .state import TrainState, LORA_TARGETS, LAYER_GROUP_DEFS',
    'from .widgets import labeled_frame',
    '',
    '_ATTENTION_ONLY_ARGS = (',
    '    "exclude_patterns=["',
    "    \"\\'.*\\\\.mlp\\\\..*\\',\\'first\\',\\'last\\\\.linear\\',\"",
    "    \"\\'tmlp\\\\..*\\',\\'txtmlp\\\\..*\\',\\'tproj\\\\.1\\',\\'txtfusion\\\\..*\\'\"",
    '    "]"',
    ')',
    '',
    '# Combobox 表示ラベル → state キーのマッピング',
    '_TARGET_LABELS: dict[str, str] = {',
    '    "全 Linear 層（264層、デフォルト・推奨）":      "all",',
    '    "Attention のみ（140層、長時間学習向け）":       "attention_only",',
    '    "カスタム":                                      "custom",',
    '}',
    '_TARGET_KEYS_TO_LABEL: dict[str, str] = {v: k for k, v in _TARGET_LABELS.items()}',
    '',
]

# ファイルの大半は複雑なエスケープを避けるため直接 write する
TARGET.write_text("\n".join(LINES), encoding="utf-8", newline="\n")

# 残りは append で書く（エスケープ問題を回避）
with TARGET.open("a", encoding="utf-8", newline="\n") as f:
    f.write('''

def build_network_tab(parent: ttk.Frame, s: TrainState) -> None:
    """ネットワークタブの UI を parent に構築する。"""
    parent.columnconfigure(1, weight=1)

    # ── LoRA 基本設定 ─────────────────────────────────────────────
    lf = labeled_frame(parent, "LoRA 設定")
    lf.columnconfigure(1, weight=1)

    ttk.Label(lf, text="network_dim", width=24, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=1024, textvariable=s.network_dim, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="network_alpha", width=24, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.network_alpha, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="network_module", width=24, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Label(lf, text="networks.lora_krea2  (固定)", foreground="#1D4ED8").grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    # ── LoRA ターゲット（Combobox）────────────────────────────────
    lf2 = labeled_frame(parent, "LoRA ターゲット層")
    lf2.columnconfigure(1, weight=1)

    ttk.Label(lf2, text="ターゲット", width=24, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=4)

    _combo_var = tk.StringVar(
        value=_TARGET_KEYS_TO_LABEL.get(s.lora_target.get(),
                                        "全 Linear 層（264層、デフォルト・推奨）")
    )
    combo = ttk.Combobox(
        lf2,
        textvariable=_combo_var,
        values=list(_TARGET_LABELS.keys()),
        state="readonly",
        width=44,
    )
    combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 4), pady=4)

    # ヒントラベル
    hint_lbl = ttk.Label(lf2, foreground="#64748B", justify=tk.LEFT, wraplength=520)
    hint_lbl.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=(4, 4), pady=2)

    # ── カスタム専用エリア（フレームで包んで show/hide）──────────
    custom_frame = ttk.Frame(lf2)
    custom_frame.columnconfigure(1, weight=1)

    # サブモードラジオ
    submode_frame = ttk.Frame(custom_frame)
    submode_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W,
                       padx=(4, 4), pady=(6, 2))
    ttk.Label(submode_frame, text="入力方法：").pack(side=tk.LEFT)
    ttk.Radiobutton(
        submode_frame, text="テキスト直接入力",
        variable=s.layer_input_mode, value="text",
        command=lambda: _refresh_custom_submode(s, text_frame, args_entry),
    ).pack(side=tk.LEFT, padx=(4, 0))
    ttk.Radiobutton(
        submode_frame, text="層グループ選択",
        variable=s.layer_input_mode, value="group",
        command=lambda: _refresh_custom_submode(s, text_frame, args_entry),
    ).pack(side=tk.LEFT, padx=(8, 0))

    # ── サブモード A: テキスト直接入力 ──────────────────────────
    text_frame = ttk.Frame(custom_frame)
    text_frame.columnconfigure(1, weight=1)
    ttk.Label(text_frame, text="network_args", width=24, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    args_entry = ttk.Entry(text_frame, textvariable=s.network_args)
    args_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 4), pady=3)
    ttk.Label(
        text_frame,
        text="例: exclude_patterns=[\\'.*\\'] include_patterns=[\\'.*wq.*\\',\\'.*wk.*\\']",
        foreground="#64748B",
    ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=(4, 4), pady=2)
    text_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW)

    # ── サブモード B: 層グループ選択 ─────────────────────────────
    group_frame = ttk.Frame(custom_frame)
    group_frame.columnconfigure(0, weight=1)
    _build_group_selector(group_frame, s, args_entry)
    group_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW)

    # ── Combobox 変更コールバック ─────────────────────────────────
    def _on_combo_change(*_) -> None:
        key = _TARGET_LABELS.get(_combo_var.get(), "all")
        s.lora_target.set(key)
        _on_target_change(s, args_entry, hint_lbl, custom_frame, lf2)

    combo.bind("<<ComboboxSelected>>", _on_combo_change)

    # 初期表示
    _on_target_change(s, args_entry, hint_lbl, custom_frame, lf2)


# ── 内部関数 ────────────────────────────────────────────────────────────────

def _build_group_selector(
    parent: ttk.Frame,
    s: TrainState,
    args_entry: ttk.Entry,
) -> None:
    """層グループ選択チェックボックス群を構築する。"""
    cb_frame = ttk.LabelFrame(parent, text="学習対象層（チェック=有効）")
    cb_frame.grid(row=0, column=0, sticky=tk.EW, padx=(4, 4), pady=(4, 2))
    cb_frame.columnconfigure(0, weight=1)

    for i, (key, label, _pat) in enumerate(LAYER_GROUP_DEFS):
        var = s.layer_groups[key]
        ttk.Checkbutton(
            cb_frame, text=label, variable=var,
            command=lambda: _sync_group_to_preview(s, args_entry),
        ).grid(row=i, column=0, sticky=tk.W, padx=(6, 4), pady=1)

    btn_frame = ttk.Frame(parent)
    btn_frame.grid(row=1, column=0, sticky=tk.W, padx=(4, 4), pady=(2, 2))

    ttk.Button(
        btn_frame, text="全 ON",
        command=lambda: _set_all_groups(s, True, args_entry),
    ).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Button(
        btn_frame, text="全 OFF",
        command=lambda: _set_all_groups(s, False, args_entry),
    ).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Button(
        btn_frame, text="プレビュー（network_args）",
        command=lambda: _show_args_preview(s),
    ).pack(side=tk.LEFT, padx=(0, 4))


def _set_all_groups(s: TrainState, value: bool, args_entry: ttk.Entry) -> None:
    """全グループを一括で ON/OFF する。"""
    for var in s.layer_groups.values():
        var.set(value)
    _sync_group_to_preview(s, args_entry)


def _sync_group_to_preview(s: TrainState, args_entry: ttk.Entry) -> None:
    """グループ選択状態を network_args StringVar に同期する（表示専用）。

    args_entry を readonly にして生成結果を表示する。
    実際の CLI 引数生成は cmd_builder._build_network_args_from_groups() が行う。
    """
    args_str = _generate_network_args(s)
    args_entry.configure(state="normal")
    s.network_args.set(args_str)
    args_entry.configure(state="readonly")


def _generate_network_args(s: TrainState) -> str:
    """layer_groups の状態から network_args 文字列を生成して返す。

    全グループ ON → 空文字（--network_args なし）。
    一部/全部 OFF → exclude_patterns=[...] 形式。
    """
    off_keys = [key for key, var in s.layer_groups.items() if not var.get()]
    if not off_keys:
        return ""

    pat_map = {key: pat for key, _label, pat in LAYER_GROUP_DEFS}
    patterns = [pat_map[k] for k in off_keys if k in pat_map]
    if not patterns:
        return ""

    quoted = ", ".join("'" + p + "'" for p in patterns)
    return "exclude_patterns=[" + quoted + "]"


def _show_args_preview(s: TrainState) -> None:
    """network_args のプレビューをポップアップ表示する。"""
    args_str = _generate_network_args(s)
    if not args_str:
        msg = "--network_args なし（全 Linear 層が対象）"
    else:
        msg = '--network_args "' + args_str + '"'
    messagebox.showinfo("network_args プレビュー", msg)


def _refresh_custom_submode(
    s: TrainState,
    text_frame: ttk.Frame,
    args_entry: ttk.Entry,
) -> None:
    """サブモード切替時に args_entry の編集可否を切り替える。"""
    if s.layer_input_mode.get() == "text":
        args_entry.configure(state="normal")
    else:
        _sync_group_to_preview(s, args_entry)


def _on_target_change(
    s: TrainState,
    args_entry: ttk.Entry,
    hint_lbl: ttk.Label,
    custom_frame: ttk.Frame,
    lf2: ttk.LabelFrame,
) -> None:
    """Combobox 変更時にヒント・カスタムフレーム表示を更新する。"""
    target = s.lora_target.get()

    if target == "all":
        custom_frame.grid_remove()
        s.network_args.set("")
        hint_lbl.configure(
            text="attention / MLP / txtfusion / projection を含む全 Linear 層（264層）を対象とします。")

    elif target == "attention_only":
        custom_frame.grid_remove()
        s.network_args.set(_ATTENTION_ONLY_ARGS)
        hint_lbl.configure(
            text="wq/wk/wv/wo/gate の 140 層のみ。"
                 "ランクを上げてプロンプト追従性を重視する場合に。")

    else:  # custom
        hint_lbl.configure(
            text="テキスト直接入力 または 層グループ選択で対象層を指定します。")
        custom_frame.grid(row=2, column=0, columnspan=2,
                          sticky=tk.EW, padx=0, pady=(4, 2))
        _refresh_custom_submode(s, None, args_entry)


def _refresh_custom_submode(  # noqa: F811  (再定義は意図的)
    s: TrainState,
    _unused: object,
    args_entry: ttk.Entry,
) -> None:
    """サブモード切替時に args_entry の編集可否を切り替える（最終版）。"""
    if s.layer_input_mode.get() == "text":
        args_entry.configure(state="normal")
    else:
        _sync_group_to_preview(s, args_entry)
''')

print(f"[OK] replaced {TARGET}")
print("\n[apply_fix_02] tab_network.py の置き換え完了。")
