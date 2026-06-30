"""app/tab_dataset.py — [データセット] タブ UI ビルダー。

musubi-tuner スキーマ準拠の toml を生成する。
sd-scripts 固有の shuffle_caption / keep_tokens / flip_aug /
min_bucket_reso / max_bucket_reso は存在しないため UI から除去済み。

モード切替:
  GUI 入力モード（デフォルト）: 入力値から configs/ に toml を自動生成。
  TOML 直接モード           : 既存 toml ファイルをブラウズして直接指定。
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .dataset_config import DatasetConfig, DatasetEntry, generate_toml, write_toml
from .state import (
    TrainState, DatasetEntryVars,
    DATASET_MODE_GUI, DATASET_MODE_TOML,
)


# ── 公開エントリポイント ─────────────────────────────────────────────────────

def build_dataset_tab(parent: ttk.Frame, s: TrainState) -> None:
    """データセットタブの UI を parent に構築する。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    # ── モード切替（row 0）──────────────────────────────────────
    mode_lf = ttk.LabelFrame(parent, text="入力モード")
    mode_lf.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))

    ttk.Radiobutton(
        mode_lf,
        text="GUI 入力（自動 toml 生成・推奨）",
        variable=s.dataset_mode, value=DATASET_MODE_GUI,
        command=lambda: _switch_mode(s, gui_area, toml_area),
    ).pack(side=tk.LEFT, padx=(8, 16), pady=4)

    ttk.Radiobutton(
        mode_lf,
        text="既存 toml ファイルを直接指定",
        variable=s.dataset_mode, value=DATASET_MODE_TOML,
        command=lambda: _switch_mode(s, gui_area, toml_area),
    ).pack(side=tk.LEFT, padx=(0, 8), pady=4)

    # ── コンテンツ（row 1）──────────────────────────────────────
    content = ttk.Frame(parent)
    content.grid(row=1, column=0, sticky=tk.NSEW)
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)

    gui_area = ttk.Frame(content)
    gui_area.grid(row=0, column=0, sticky=tk.NSEW)
    gui_area.columnconfigure(0, weight=1)
    gui_area.rowconfigure(1, weight=1)

    toml_area = ttk.Frame(content)
    toml_area.grid(row=0, column=0, sticky=tk.NSEW)
    toml_area.columnconfigure(0, weight=1)

    _build_gui_mode(gui_area, s, gui_area, toml_area)
    _build_toml_mode(toml_area, s)
    _switch_mode(s, gui_area, toml_area)


# ── GUI 入力モード ───────────────────────────────────────────────────────────

def _build_gui_mode(
    parent: ttk.Frame,
    s: TrainState,
    gui_area: "ttk.Frame",
    toml_area: "ttk.Frame",
) -> None:
    """GUI 入力モードの UI を parent に構築する（全 grid）。"""

    # ── [general] 共通設定（row 0）───────────────────────────────
    common_lf = ttk.LabelFrame(parent, text="[general] 共通設定")
    common_lf.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    common_lf.columnconfigure(1, weight=0)
    common_lf.columnconfigure(3, weight=0)
    common_lf.columnconfigure(5, weight=0)

    ttk.Label(common_lf, text="resolution", width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(common_lf, textvariable=s.general_resolution,
                 values=[512, 768, 1024, 1280], width=7, state="normal").grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)

    ttk.Label(common_lf, text="caption_ext", width=14, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(common_lf, textvariable=s.general_caption_extension,
                 values=[".txt", ".caption", ".tag"],
                 width=10, state="normal").grid(
        row=0, column=3, sticky=tk.W, padx=(0, 12), pady=3)

    ttk.Label(common_lf, text="batch_size", width=12, anchor=tk.W).grid(
        row=0, column=4, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(common_lf, from_=1, to=64, textvariable=s.batch_size,
                width=6).grid(row=0, column=5, sticky=tk.W, padx=(0, 4), pady=3)

    chk_row = ttk.Frame(common_lf)
    chk_row.grid(row=1, column=0, columnspan=6, sticky=tk.W, padx=(4, 4), pady=3)
    ttk.Checkbutton(chk_row, text="enable_bucket",
                    variable=s.general_enable_bucket).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Checkbutton(chk_row, text="bucket_no_upscale",
                    variable=s.general_bucket_no_upscale).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Label(chk_row, text="num_workers", anchor=tk.W).pack(side=tk.LEFT)
    ttk.Spinbox(chk_row, from_=0, to=32, textvariable=s.num_workers,
                width=5).pack(side=tk.LEFT, padx=(2, 8))
    ttk.Checkbutton(chk_row, text="persistent_workers",
                    variable=s.persistent_workers).pack(side=tk.LEFT)

    # ── [[datasets]] エントリスクロール領域（row 1）─────────────
    scroll_lf = ttk.LabelFrame(parent, text="[[datasets]] エントリ")
    scroll_lf.grid(row=1, column=0, sticky=tk.NSEW, pady=(0, 4))
    scroll_lf.columnconfigure(0, weight=1)
    scroll_lf.rowconfigure(0, weight=1)

    canvas = tk.Canvas(scroll_lf, borderwidth=0, highlightthickness=0)
    vsb = ttk.Scrollbar(scroll_lf, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.grid(row=0, column=1, sticky=tk.NS)
    canvas.grid(row=0, column=0, sticky=tk.NSEW)

    inner = ttk.Frame(canvas)
    inner.columnconfigure(0, weight=1)
    inner_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

    def _on_inner_configure(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event) -> None:
        canvas.itemconfig(inner_id, width=event.width)

    inner.bind("<Configure>", _on_inner_configure)
    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    # ── ボタン行（row 2）─────────────────────────────────────────
    btn_row = ttk.Frame(parent)
    btn_row.grid(row=2, column=0, sticky=tk.EW, pady=(0, 2))

    def _rebuild() -> None:
        for w in inner.winfo_children():
            w.destroy()
        for idx, ev in enumerate(s.dataset_entries):
            card = _build_entry_card(inner, s, ev, idx, _rebuild)
            card.grid(row=idx, column=0, sticky=tk.EW, padx=4, pady=(4, 0))
        _on_inner_configure()

    def _add_entry() -> None:
        s.dataset_entries.append(DatasetEntryVars())
        _rebuild()

    def _preview_toml() -> None:
        try:
            toml_str = _state_to_toml(s)
            _show_preview(parent, toml_str)
        except Exception as exc:
            messagebox.showerror("プレビューエラー", str(exc))

    ttk.Button(btn_row, text="+ エントリ追加",
               command=_add_entry).pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_row, text="toml プレビュー",
               command=_preview_toml).pack(side=tk.LEFT, padx=4)

    _rebuild()

    def _on_preset_reload() -> None:
        """プリセット読み込み後: カード再描画 + モード切替 UI 同期。"""
        _rebuild()
        _switch_mode(s, gui_area, toml_area)

    # プリセット読み込み時に再描画されるよう TrainState にコールバック登録
    s.register_dataset_reload_callback(_on_preset_reload)


def _build_entry_card(
    parent: ttk.Frame,
    s: TrainState,
    ev: DatasetEntryVars,
    idx: int,
    rebuild_fn,
) -> ttk.LabelFrame:
    """1 [[datasets]] エントリのカード UI（musubi-tuner スキーマ準拠）。"""
    card = ttk.LabelFrame(parent, text=f"データセット {idx + 1}")
    card.columnconfigure(1, weight=1)

    # 画像ディレクトリ（必須）
    ttk.Label(card, text="image_directory *", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(card, textvariable=ev.image_dir).grid(
        row=0, column=1, sticky=tk.EW, padx=(0, 2), pady=3)
    ttk.Button(card, text="Browse", width=7,
               command=lambda v=ev.image_dir: _pick_dir(v)).grid(
        row=0, column=2, padx=(0, 4), pady=3)

    # キャッシュディレクトリ（推奨）
    ttk.Label(card, text="cache_directory", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(card, textvariable=ev.cache_directory).grid(
        row=1, column=1, sticky=tk.EW, padx=(0, 2), pady=3)
    ttk.Button(card, text="Browse", width=7,
               command=lambda v=ev.cache_directory: _pick_dir(v)).grid(
        row=1, column=2, padx=(0, 4), pady=3)

    # num_repeats / caption_extension（general 上書き用）
    r2 = ttk.Frame(card)
    r2.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=(4, 4), pady=3)
    ttk.Label(r2, text="num_repeats", width=16, anchor=tk.W).pack(side=tk.LEFT)
    ttk.Spinbox(r2, from_=1, to=9999, textvariable=ev.num_repeats,
                width=7).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Label(r2, text="caption_ext（上書き）", width=20, anchor=tk.W).pack(side=tk.LEFT)
    ttk.Combobox(r2, textvariable=ev.caption_extension,
                 values=[".txt", ".caption", ".tag"],
                 width=10, state="normal").pack(side=tk.LEFT, padx=(0, 16))
    ttk.Label(r2, text="resolution（上書き）", width=18, anchor=tk.W).pack(side=tk.LEFT)
    ttk.Combobox(r2, textvariable=ev.resolution,
                 values=[512, 768, 1024, 1280],
                 width=7, state="normal").pack(side=tk.LEFT)

    # 削除ボタン（2件以上）
    if len(s.dataset_entries) > 1:
        def _remove(i=idx):
            s.dataset_entries.pop(i)
            rebuild_fn()
        ttk.Button(card, text="✕ このデータセットを削除",
                   command=_remove).grid(
            row=3, column=0, columnspan=3, sticky=tk.E,
            padx=(0, 4), pady=(2, 4))

    return card


# ── TOML 直接モード ──────────────────────────────────────────────────────────

def _build_toml_mode(parent: ttk.Frame, s: TrainState) -> None:
    """既存 toml 直接指定モードの UI（全 grid）。"""
    lf = ttk.LabelFrame(parent, text="既存 dataset_config.toml を直接指定")
    lf.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    lf.columnconfigure(1, weight=1)

    ttk.Label(lf, text="toml ファイル", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.dataset_config_path).grid(
        row=0, column=1, sticky=tk.EW, padx=(0, 2), pady=3)
    ttk.Button(lf, text="Browse", width=7,
               command=lambda: _pick_toml(s)).grid(
        row=0, column=2, padx=(0, 4), pady=3)

    ttk.Label(
        parent,
        text="musubi-tuner の image_directory 形式の toml を指定してください。\n"
             "（sd-scripts の [[datasets.subsets]] 形式は使用できません）",
        foreground="#64748B",
        justify=tk.LEFT,
    ).grid(row=1, column=0, sticky=tk.W, padx=6, pady=(4, 0))


# ── モード切替 ───────────────────────────────────────────────────────────────

def _switch_mode(
    s: TrainState,
    gui_area: ttk.Frame,
    toml_area: ttk.Frame,
) -> None:
    if s.dataset_mode.get() == DATASET_MODE_GUI:
        toml_area.lower()
        gui_area.lift()
    else:
        gui_area.lower()
        toml_area.lift()


# ── dataset_config パス解決（cmd_builder から呼ばれる）──────────────────────

def resolve_dataset_config(s: TrainState) -> str:
    """学習コマンド用の dataset_config パスを解決して返す。"""
    if s.dataset_mode.get() == DATASET_MODE_TOML:
        p = s.dataset_config_path.get().strip()
        if not p:
            raise ValueError("toml ファイルパスが未指定です。")
        if not Path(p).exists():
            raise ValueError(f"toml ファイルが見つかりません: {p}")
        return p

    cfg  = _state_to_cfg(s)
    dest = write_toml(cfg, s.paths.root / "configs",
                      stem=s.output_name.get() or "dataset")
    return str(dest)


# ── 内部ヘルパー ─────────────────────────────────────────────────────────────

def _state_to_cfg(s: TrainState) -> DatasetConfig:
    entries = [ev.to_entry() for ev in s.dataset_entries
               if ev.image_dir.get().strip()]
    if not entries:
        raise ValueError("image_directory が1件も設定されていません。")
    return DatasetConfig(
        resolution        = s.general_resolution.get(),
        caption_extension = s.general_caption_extension.get(),
        batch_size        = s.batch_size.get(),
        enable_bucket     = s.general_enable_bucket.get(),
        bucket_no_upscale = s.general_bucket_no_upscale.get(),
        entries           = entries,
    )


def _state_to_toml(s: TrainState) -> str:
    return generate_toml(_state_to_cfg(s))


def _pick_dir(var: tk.StringVar) -> None:
    path = filedialog.askdirectory()
    if path:
        var.set(path)


def _pick_toml(s: TrainState) -> None:
    path = filedialog.askopenfilename(
        filetypes=[("TOML", "*.toml"), ("All", "*.*")])
    if path:
        s.dataset_config_path.set(path)


def _show_preview(parent: tk.Widget, toml_str: str) -> None:
    dlg = tk.Toplevel(parent)
    dlg.title("dataset_config.toml プレビュー")
    dlg.geometry("700x500")
    dlg.grab_set()
    text = tk.Text(dlg, wrap=tk.NONE, font=("TkFixedFont", 9))
    xsb  = ttk.Scrollbar(dlg, orient=tk.HORIZONTAL, command=text.xview)
    ysb  = ttk.Scrollbar(dlg, orient=tk.VERTICAL,   command=text.yview)
    text.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
    ysb.pack(side=tk.RIGHT,  fill=tk.Y)
    xsb.pack(side=tk.BOTTOM, fill=tk.X)
    text.pack(fill=tk.BOTH,  expand=True)
    text.insert("1.0", toml_str)
    text.configure(state=tk.DISABLED)
    ttk.Button(dlg, text="閉じる", command=dlg.destroy).pack(pady=4)
