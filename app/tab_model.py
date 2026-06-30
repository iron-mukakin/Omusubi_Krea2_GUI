"""app/tab_model.py — [モデル] タブ UI ビルダー。

parent への配置は pack のみ（labeled_frame 経由）。
LabelFrame 内の子ウィジェットは grid のみ。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .state import TrainState, PRECISIONS
from .widgets import browse_file_row, browse_dir_row, label_combobox_row, labeled_frame


def build_model_tab(parent: ttk.Frame, s: TrainState) -> None:
    """モデルパスタブの UI を parent に構築する。"""

    # ── モデルパス ──────────────────────────────────────────────
    lf = labeled_frame(parent, "モデルパス")    # lf は pack 済み
    lf.columnconfigure(1, weight=1)

    browse_file_row(lf, 0, "DiT (RAW) *",  s.dit_path)
    browse_file_row(lf, 1, "VAE *",         s.vae_path)
    browse_file_row(lf, 2, "Text Encoder",  s.text_encoder_path,
                   filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
    # turbo_dit / turbo_dit_cache は [詳細・最適化] タブで設定

    # ── 出力設定 ─────────────────────────────────────────────────
    lf2 = labeled_frame(parent, "出力設定")    # lf2 は pack 済み
    lf2.columnconfigure(1, weight=1)

    browse_dir_row(lf2, 0, "出力フォルダ *", s.output_dir)

    ttk.Label(lf2, text="出力ファイル名", width=24, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.output_name).grid(
        row=1, column=1, sticky=tk.EW, padx=(0, 4), pady=3)

    label_combobox_row(lf2, 2, "save_precision", s.save_precision,
                       PRECISIONS, combo_width=10)
    browse_file_row(lf2, 3, "network_weights", s.network_weights)

    # ── 注記（pack で parent に追加）───────────────────────────
    ttk.Label(
        parent,
        text="* 必須。Text Encoder はサンプル生成時のみ必要。",
        foreground="#64748B",
    ).pack(anchor=tk.W, padx=6, pady=(2, 0))
