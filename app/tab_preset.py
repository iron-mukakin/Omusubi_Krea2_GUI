"""app/tab_preset.py — [プリセット] タブ UI ビルダー。"""
from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .state import TrainState
from . import preset_manager


def build_preset_tab(parent: ttk.Frame, s: TrainState) -> None:
    """プリセットタブの UI を parent に構築する。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    preset_dir = s.paths.preset / "krea2"

    # ── リストボックス ────────────────────────────────────────────
    list_frame = ttk.LabelFrame(parent, text="保存済みプリセット")
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

    lb = tk.Listbox(list_frame, height=12, selectmode=tk.SINGLE,
                    font=("TkDefaultFont", 10))
    lb_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=lb.yview)
    lb.configure(yscrollcommand=lb_scroll.set)
    lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── 名前入力 ─────────────────────────────────────────────────
    name_row = ttk.Frame(parent)
    name_row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(name_row, text="プリセット名:").pack(side=tk.LEFT)
    name_var = tk.StringVar()
    ttk.Entry(name_row, textvariable=name_var, width=32).pack(
        side=tk.LEFT, padx=(4, 0))

    # ── ボタン行 ─────────────────────────────────────────────────
    btn_row = ttk.Frame(parent)
    btn_row.pack(fill=tk.X)

    def _refresh() -> None:
        lb.delete(0, tk.END)
        for name in preset_manager.list_presets(preset_dir):
            lb.insert(tk.END, name)

    def _save() -> None:
        name = name_var.get().strip()
        if not name:
            messagebox.showerror("プリセット", "プリセット名を入力してください。")
            return
        err = preset_manager.save_to_file(s, preset_dir, name)
        if err:
            messagebox.showerror("プリセット", err)
            return
        _refresh()
        s.log_fn(f"[PRESET] 保存: {name}")

    def _load() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror("プリセット", "プリセットを選択してください。")
            return
        path = preset_dir / f"{lb.get(sel[0])}.json"
        err = preset_manager.load_from_file(s, path)
        if err:
            messagebox.showerror("プリセット", err)
            return
        name_var.set(path.stem)
        s.log_fn(f"[PRESET] 読込: {path.name}")

    def _delete() -> None:
        sel = lb.curselection()
        if not sel:
            return
        pname = lb.get(sel[0])
        if not messagebox.askyesno("プリセット", f"「{pname}」を削除しますか？"):
            return
        (preset_dir / f"{pname}.json").unlink(missing_ok=True)
        _refresh()
        s.log_fn(f"[PRESET] 削除: {pname}.json")

    def _export() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror("プリセット", "プリセットを選択してください。")
            return
        src = preset_dir / f"{lb.get(sel[0])}.json"
        dest = filedialog.asksaveasfilename(
            initialdir=str(preset_dir),
            initialfile=src.name,
            filetypes=[("JSON", "*.json")],
        )
        if dest:
            shutil.copy2(src, dest)
            s.log_fn(f"[PRESET] エクスポート: {dest}")

    def _import() -> None:
        src = filedialog.askopenfilename(
            initialdir=str(preset_dir),
            filetypes=[("JSON", "*.json")],
        )
        if not src:
            return
        dest = preset_dir / Path(src).name
        preset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        _refresh()
        s.log_fn(f"[PRESET] インポート: {Path(src).name}")

    for text, cmd in [
        ("保存",           _save),
        ("読込",           _load),
        ("削除",           _delete),
        ("エクスポート",    _export),
        ("インポート",      _import),
        ("一覧更新",        _refresh),
    ]:
        ttk.Button(btn_row, text=text, command=cmd).pack(
            side=tk.LEFT, padx=4, pady=4)

    _refresh()
