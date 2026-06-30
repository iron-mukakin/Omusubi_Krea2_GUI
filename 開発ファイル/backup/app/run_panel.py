"""app/run_panel.py — 実行パネルビルダー。

コマンドプレビュー・開始・停止ボタン・ステータス・ログ出力を
各メインタブの下部に共通で配置する。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .state import TrainState
from .cmd_builder import build_train_command, validate_inputs
from .runner import launch_training, stop_training, start_log_drain
from .widgets import attach_log_widget


def build_run_panel(
    parent: ttk.Frame,
    s: TrainState,
    root: tk.Tk,
) -> None:
    """実行パネルを parent の下部に pack する。"""
    frm = ttk.LabelFrame(parent, text="実行")
    frm.pack(fill=tk.X, pady=(6, 0))

    # ── コマンドプレビュー ───────────────────────────────────────
    preview_row = ttk.Frame(frm)
    preview_row.pack(fill=tk.X, padx=4, pady=(4, 0))
    ttk.Label(preview_row, text="コマンドプレビュー:").pack(side=tk.LEFT)
    ttk.Button(
        preview_row, text="更新",
        command=lambda: _refresh_preview(s, cmd_text),
    ).pack(side=tk.LEFT, padx=4)

    cmd_text = tk.Text(frm, height=3, wrap=tk.WORD, font=("TkFixedFont", 8),
                       state=tk.DISABLED)
    cmd_text.pack(fill=tk.X, padx=4, pady=2)

    # ── ステータス / ボタン ──────────────────────────────────────
    btn_row = ttk.Frame(frm)
    btn_row.pack(fill=tk.X, padx=4, pady=(2, 4))
    ttk.Label(btn_row, textvariable=s.status_var, foreground="#334155").pack(
        side=tk.LEFT, padx=4)

    ttk.Button(
        btn_row, text="■ 停止",
        command=lambda: stop_training(s),
    ).pack(side=tk.RIGHT, padx=(4, 0))

    ttk.Button(
        btn_row, text="▶ 学習開始",
        command=lambda: _on_start(s, cmd_text),
    ).pack(side=tk.RIGHT, padx=4)

    # ── ログ出力 ─────────────────────────────────────────────────
    log_frame = ttk.LabelFrame(parent, text="学習ログ")
    log_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
    attach_log_widget(log_frame, s._log_widgets, height=10)

    # ログドレイン開始（1 回だけ）
    start_log_drain(lambda: s._log_widgets, s, root)


def _refresh_preview(s: TrainState, cmd_text: tk.Text) -> None:
    """コマンドプレビューを更新する。"""
    cmd_text.configure(state=tk.NORMAL)
    cmd_text.delete("1.0", tk.END)
    try:
        cmd = build_train_command(s, s.paths)
        cmd_text.insert(tk.END, " ".join(cmd))
    except Exception as exc:
        cmd_text.insert(tk.END, f"[コマンド生成エラー] {exc}")
    cmd_text.configure(state=tk.DISABLED)


def _on_start(s: TrainState, cmd_text: tk.Text) -> None:
    """バリデーション → コマンド生成 → 学習起動。"""
    if s.is_running():
        messagebox.showwarning("学習中", "学習が既に実行中です。")
        return

    err = validate_inputs(s)
    if err:
        messagebox.showerror("入力エラー", err)
        return

    # musubi-tuner の存在確認
    musubi_err = s.paths.validate_musubi()
    if musubi_err:
        messagebox.showerror("セットアップエラー", musubi_err)
        return

    try:
        cmd = build_train_command(s, s.paths)
    except Exception as exc:
        messagebox.showerror("コマンド生成エラー", str(exc))
        return

    _refresh_preview(s, cmd_text)
    s.paths.output.mkdir(parents=True, exist_ok=True)

    launch_training(s, cmd)
