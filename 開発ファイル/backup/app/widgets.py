"""app/widgets.py — 再利用可能な Tk ウィジェットヘルパー群。

各タブビルダーから import して使用する。
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable


def label_entry_row(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.Variable,
    width: int = 24,
    entry_width: int = 0,
) -> ttk.Entry:
    """ラベル + Entry を grid 1行に配置して Entry を返す。"""
    ttk.Label(parent, text=label, width=width, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3,
    )
    kw = {"textvariable": var}
    if entry_width:
        kw["width"] = entry_width
    entry = ttk.Entry(parent, **kw)
    entry.grid(row=row, column=1, sticky=tk.EW, padx=(0, 4), pady=3)
    return entry


def label_spinbox_row(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.Variable,
    from_: float,
    to: float,
    increment: float = 1,
    width: int = 24,
    spin_width: int = 8,
) -> ttk.Spinbox:
    """ラベル + Spinbox を grid 1行に配置して Spinbox を返す。"""
    ttk.Label(parent, text=label, width=width, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3,
    )
    sb = ttk.Spinbox(
        parent, from_=from_, to=to, increment=increment,
        textvariable=var, width=spin_width,
    )
    sb.grid(row=row, column=1, sticky=tk.W, padx=(0, 4), pady=3)
    return sb


def label_combobox_row(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.StringVar,
    values: tuple[str, ...],
    width: int = 24,
    combo_width: int = 16,
) -> ttk.Combobox:
    """ラベル + readonly Combobox を grid 1行に配置して Combobox を返す。"""
    ttk.Label(parent, text=label, width=width, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3,
    )
    cb = ttk.Combobox(
        parent, textvariable=var, values=list(values),
        state="readonly", width=combo_width,
    )
    cb.grid(row=row, column=1, sticky=tk.W, padx=(0, 4), pady=3)
    return cb


def browse_file_row(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.StringVar,
    filetypes: list[tuple[str, str]] | None = None,
    width: int = 24,
) -> None:
    """ラベル + Entry + Browse ボタンを grid 1行に配置する。"""
    ft = filetypes or [("safetensors", "*.safetensors"), ("All", "*.*")]
    ttk.Label(parent, text=label, width=width, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3,
    )
    ttk.Entry(parent, textvariable=var).grid(
        row=row, column=1, sticky=tk.EW, padx=(0, 2), pady=3,
    )
    ttk.Button(
        parent, text="Browse", width=7,
        command=lambda v=var, f=ft: _pick_file(v, f),
    ).grid(row=row, column=2, padx=(0, 4), pady=3)


def browse_dir_row(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.StringVar,
    width: int = 24,
) -> None:
    """ラベル + Entry + Browse ボタン（ディレクトリ選択）を grid 1行に配置する。"""
    ttk.Label(parent, text=label, width=width, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3,
    )
    ttk.Entry(parent, textvariable=var).grid(
        row=row, column=1, sticky=tk.EW, padx=(0, 2), pady=3,
    )
    ttk.Button(
        parent, text="Browse", width=7,
        command=lambda v=var: _pick_dir(v),
    ).grid(row=row, column=2, padx=(0, 4), pady=3)


def two_col_row(
    parent: tk.Widget,
    row: int,
    label1: str, widget1_fn: Callable[[tk.Widget], tk.Widget],
    label2: str, widget2_fn: Callable[[tk.Widget], tk.Widget],
    lw: int = 18,
) -> tuple[tk.Widget, tk.Widget]:
    """左右 2ペアのラベル+ウィジェットを 1行に配置する。"""
    ttk.Label(parent, text=label1, width=lw, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3,
    )
    w1 = widget1_fn(parent)
    w1.grid(row=row, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(parent, text=label2, width=lw, anchor=tk.W).grid(
        row=row, column=2, sticky=tk.W, padx=(0, 2), pady=3,
    )
    w2 = widget2_fn(parent)
    w2.grid(row=row, column=3, sticky=tk.W, padx=(0, 4), pady=3)
    return w1, w2


def labeled_frame(parent: tk.Widget, text: str, **pack_kw) -> ttk.LabelFrame:
    """pack 済みの LabelFrame を返す。"""
    lf = ttk.LabelFrame(parent, text=text)
    lf.pack(**{"fill": tk.X, "pady": (0, 6), **pack_kw})
    return lf


def attach_log_widget(
    parent: tk.Widget,
    log_widgets: list[tk.Text],
    height: int = 8,
) -> tk.Text:
    """スクロール付き読み取り専用ログ Text を parent に pack して返す。"""
    log_text = tk.Text(
        parent, height=height, wrap=tk.WORD,
        font=("TkFixedFont", 9), state=tk.DISABLED,
    )
    scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.BOTH, expand=True)
    log_widgets.append(log_text)
    return log_text


# ── 内部ヘルパー ────────────────────────────────────────────────────────────
def _pick_file(var: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        var.set(path)


def _pick_dir(var: tk.StringVar) -> None:
    path = filedialog.askdirectory()
    if path:
        var.set(path)
