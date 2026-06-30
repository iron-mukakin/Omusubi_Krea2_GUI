"""app/tab_sample.py — [サンプル生成] タブ UI ビルダー。"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .state import TrainState, SAMPLE_FIXED_SEED
from .widgets import labeled_frame


def build_sample_tab(parent: ttk.Frame, s: TrainState) -> None:
    """サンプル生成タブの UI を parent に構築する。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    # ── 共通設定 ─────────────────────────────────────────────────
    lf = labeled_frame(parent, "共通設定")
    lf.columnconfigure(1, weight=1)
    lf.columnconfigure(3, weight=1)

    ttk.Label(lf, text="sample_every_n_epochs", width=24, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=9999, textvariable=s.sample_every_n_epochs,
                width=8).grid(row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(
        lf,
        text=f"固定シード A={SAMPLE_FIXED_SEED}  B={SAMPLE_FIXED_SEED + 1}",
        foreground="#64748B",
    ).grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    # width / height / steps / guidance_scale
    ttk.Label(lf, text="width / height", width=24, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    wh_frame = ttk.Frame(lf)
    wh_frame.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Spinbox(wh_frame, from_=64, to=4096, increment=64,
                textvariable=s.sample_width, width=7).pack(side=tk.LEFT)
    ttk.Label(wh_frame, text=" x ").pack(side=tk.LEFT)
    ttk.Spinbox(wh_frame, from_=64, to=4096, increment=64,
                textvariable=s.sample_height, width=7).pack(side=tk.LEFT)

    ttk.Label(lf, text="steps", width=14, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=200, textvariable=s.sample_steps, width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="guidance_scale", width=24, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.sample_guidance_scale, width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)
    ttk.Label(
        lf,
        text="RAW: 5.5 推奨 / Turbo: 1（CFG オフ）",
        foreground="#64748B",
    ).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    # ── プロンプト A / B ─────────────────────────────────────────
    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    tab_a = ttk.Frame(nb, padding=6)
    tab_b = ttk.Frame(nb, padding=6)
    nb.add(tab_a, text="サンプル A")
    nb.add(tab_b, text="サンプル B")

    sample_dir_a = s.paths.log / "sample_gen" / "a"
    sample_dir_b = s.paths.log / "sample_gen" / "b"

    _build_prompt_panel(tab_a, s,
                        s.sample_enabled, s.sample_prompt,
                        s.sample_negative_prompt, sample_dir_a, "A")
    _build_prompt_panel(tab_b, s,
                        s.sample_b_enabled, s.sample_b_prompt,
                        s.sample_b_negative_prompt, sample_dir_b, "B")


def _build_prompt_panel(
    parent: ttk.Frame,
    s: TrainState,
    enabled_var: tk.BooleanVar,
    prompt_var: tk.StringVar,
    neg_var: tk.StringVar,
    sample_dir: Path,
    label: str,
) -> None:
    """1 つのサンプルプロンプトパネルを parent に構築する。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    top = ttk.Frame(parent)
    top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    top.columnconfigure(1, weight=1)

    ttk.Checkbutton(top, text=f"サンプル {label} を有効にする",
                    variable=enabled_var).grid(
        row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=2)

    ttk.Label(top, text="保存先:", foreground="#475569").grid(
        row=1, column=0, sticky=tk.W, padx=(2, 0), pady=2)
    ttk.Label(top, text=str(sample_dir), foreground="#1D4ED8").grid(
        row=1, column=1, columnspan=2, sticky=tk.W, pady=2)

    ttk.Label(top, text="prompt", width=14, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(2, 2), pady=2)
    ttk.Entry(top, textvariable=prompt_var).grid(
        row=2, column=1, columnspan=2, sticky=tk.EW, padx=(0, 4), pady=2)

    ttk.Label(top, text="negative", width=14, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(2, 2), pady=2)
    ttk.Entry(top, textvariable=neg_var).grid(
        row=3, column=1, columnspan=2, sticky=tk.EW, padx=(0, 4), pady=2)

    # ── ギャラリー ────────────────────────────────────────────────
    gallery = ttk.LabelFrame(parent, text=f"生成画像ギャラリー {label}（最新10枚）")
    gallery.grid(row=1, column=0, sticky=tk.NSEW)
    for c in range(5):
        gallery.columnconfigure(c, weight=1, uniform=f"gc_{label}")
    for r in range(2):
        gallery.rowconfigure(r, weight=1, uniform=f"gr_{label}")

    cells: list[tuple[ttk.Label, ttk.Label]] = []
    photo_refs: list = [None] * 10

    for idx in range(10):
        cell = ttk.Frame(gallery, padding=3)
        cell.grid(row=idx // 5, column=idx % 5, sticky=tk.NSEW)
        cell.columnconfigure(0, weight=1)
        cell.rowconfigure(0, weight=1)
        img_lbl = ttk.Label(cell, anchor=tk.CENTER)
        img_lbl.grid(row=0, column=0, sticky=tk.NSEW)
        ep_lbl = ttk.Label(cell, text="epoch -", anchor=tk.CENTER)
        ep_lbl.grid(row=1, column=0, sticky=tk.EW, pady=(2, 0))
        cells.append((img_lbl, ep_lbl))

    def _refresh_gallery(schedule_next: bool = False) -> None:
        files = _collect_images(sample_dir)
        try:
            from PIL import Image as _Im, ImageTk as _ITk
            pil_ok = True
        except ImportError:
            pil_ok = False

        for idx, (il, el) in enumerate(cells):
            if idx >= len(files):
                il.configure(image="", text="")
                el.configure(text="epoch -")
                photo_refs[idx] = None
                continue
            p = files[idx]
            el.configure(text=f"epoch {_parse_epoch(p)}")
            if not pil_ok:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None
                continue
            try:
                from PIL import Image as _Im, ImageTk as _ITk
                with _Im.open(p) as im:
                    im.thumbnail((200, 200))
                    ph = _ITk.PhotoImage(im.copy())
                photo_refs[idx] = ph
                il.configure(image=ph, text="")
            except Exception:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None

        if schedule_next:
            parent.after(3000, lambda: _refresh_gallery(True))

    btn_row = ttk.Frame(top)
    btn_row.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))
    ttk.Button(btn_row, text="ギャラリー更新",
               command=lambda: _refresh_gallery(False)).pack(side=tk.LEFT, padx=(0, 6))

    _refresh_gallery(True)


def _collect_images(sample_dir: Path) -> list[Path]:
    """sample_dir から PNG ファイルを更新時刻降順で最大 10 件返す。"""
    if not sample_dir.exists():
        return []
    return sorted(
        sample_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:10]


def _parse_epoch(path: Path) -> str:
    """ファイル名からエポック番号を抽出する。"""
    import re
    m = re.search(r"(?:^|_)e?(\d+)(?:_|$)", path.stem)
    return str(int(m.group(1))) if m else "-"
