"""app/tab_train.py — [学習設定] タブ UI ビルダー。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .state import TrainState, OPTIMIZERS, LR_SCHEDULERS, PRECISIONS
from .widgets import labeled_frame, two_col_row


def build_train_tab(parent: ttk.Frame, s: TrainState) -> None:
    """学習設定タブの UI を parent に構築する。"""
    parent.columnconfigure(1, weight=1)
    parent.columnconfigure(3, weight=1)

    lf = labeled_frame(parent, "学習率 / オプティマイザ")
    lf.columnconfigure(1, weight=1)
    lf.columnconfigure(3, weight=1)

    # row 0: learning_rate / lr_scheduler
    ttk.Label(lf, text="learning_rate", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.learning_rate, width=12).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="lr_scheduler", width=18, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.lr_scheduler, values=list(LR_SCHEDULERS),
                 state="readonly", width=20).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 1: lr_warmup_steps / optimizer_type
    ttk.Label(lf, text="lr_warmup_steps", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=0, to=100000, textvariable=s.lr_warmup_steps, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="optimizer_type", width=18, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.optimizer_type, values=list(OPTIMIZERS),
                 state="readonly", width=20).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: optimizer_args (full width)
    ttk.Label(lf, text="optimizer_args", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.optimizer_args).grid(
        row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=3)

    # ── エポック設定 ─────────────────────────────────────────────
    lf2 = labeled_frame(parent, "エポック / バッチ")
    lf2.columnconfigure(1, weight=1)
    lf2.columnconfigure(3, weight=1)

    # row 0: max_train_epochs / save_every_n_epochs
    ttk.Label(lf2, text="max_train_epochs", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=9999, textvariable=s.max_train_epochs, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="save_every_n_epochs", width=18, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=9999, textvariable=s.save_every_n_epochs, width=8).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 1: seed / grad_accum
    ttk.Label(lf2, text="seed", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.seed, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="gradient_accumulation", width=18, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=256, textvariable=s.grad_accum, width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: mixed_precision / max_grad_norm
    ttk.Label(lf2, text="mixed_precision", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.mixed_precision, values=list(PRECISIONS),
                 state="readonly", width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="max_grad_norm", width=18, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.max_grad_norm, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # ── フラグ ───────────────────────────────────────────────────
    lf3 = labeled_frame(parent, "チェックオプション")
    ttk.Checkbutton(lf3, text="gradient_checkpointing",
                    variable=s.gradient_checkpointing).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)
