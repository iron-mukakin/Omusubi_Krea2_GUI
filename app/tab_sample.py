"""app/tab_sample.py — [サンプル生成] タブ UI ビルダー。"""
from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

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

    # musubi-tuner (trainer_base.py) はサンプル画像の保存先を
    # "<output_dir>/sample" に固定している。GUI 側も同ディレクトリを直接
    # 参照し、複製処理は行わない。output_dir は実行中に変更されうる
    # tk.StringVar のため、固定 Path として一度だけ捕捉せず、
    # _resolve_sample_dir() で参照の都度動的に解決する。

    render_a = _build_prompt_panel(tab_a, s,
                        s.sample_enabled, s.sample_prompt,
                        s.sample_negative_prompt,
                        SAMPLE_FIXED_SEED, "A")
    render_b = _build_prompt_panel(tab_b, s,
                        s.sample_b_enabled, s.sample_b_prompt,
                        s.sample_b_negative_prompt,
                        SAMPLE_FIXED_SEED + 1, "B")

    def _poll_sample_outputs() -> None:
        """両パネルのギャラリーを再描画する。

        3秒間隔で自己再スケジュールする（tab_sample.py の生存期間中、常時実行）。
        画像は output_dir/sample に直接生成されるため複製処理は不要で、
        ここでは再描画のみを行う。
        """
        render_a()
        render_b()
        parent.after(3000, _poll_sample_outputs)

    parent.after(100, _poll_sample_outputs)


def _build_prompt_panel(
    parent: ttk.Frame,
    s: TrainState,
    enabled_var: tk.BooleanVar,
    prompt_var: tk.StringVar,
    neg_var: tk.StringVar,
    seed: int,
    label: str,
) -> Callable[[], None]:
    """1 つのサンプルプロンプトパネルを parent に構築し、ギャラリー再描画関数を返す。

    表示対象ディレクトリ (output_dir/sample) は、s.output_dir (tk.StringVar) から
    再描画のたびに _resolve_sample_dir() で動的に解決する。構築時に固定 Path として
    キャプチャすると、実行中に output_dir が変更された場合に古い参照を
    見続ける不具合となるため、意図的にこの設計としている。

    ディレクトリ内の PNG のうち、ファイル名末尾の seed サフィックスが
    引数 seed と一致するものだけを表示対象とする。musubi-tuner は
    サンプル A/B を同一フォルダへ出力し、プロンプトファイル内の
    出現順 (enum) は A/B いずれか一方のみ有効時に破綻するため、
    固定シードによる判別が唯一の信頼できる手段となる。

    戻り値の Callable はスケジューリングを持たない再描画専用関数であり、
    呼び出し元 (build_sample_tab の _poll_sample_outputs) が
    再描画のタイミングを一元管理する。
    """
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
    dest_label = ttk.Label(top, foreground="#1D4ED8")
    dest_label.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=2)

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

    def _render_gallery() -> None:
        """output_dir/sample 内の該当 seed 画像を最大10枚ギャラリーへ反映する（スケジューリングなし）。"""
        sample_dir = _resolve_sample_dir(s)
        dest_label.configure(text=f"{sample_dir} (seed={seed} で判別)")
        files = _collect_images(sample_dir, seed)
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
                with _Im.open(p) as im:
                    im.thumbnail((200, 200))
                    ph = _ITk.PhotoImage(im.copy())
                photo_refs[idx] = ph
                il.configure(image=ph, text="")
            except Exception:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None

    btn_row = ttk.Frame(top)
    btn_row.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))
    ttk.Button(btn_row, text="ギャラリー更新",
               command=_render_gallery).pack(side=tk.LEFT, padx=(0, 6))

    _render_gallery()
    return _render_gallery


# musubi-tuner (trainer_base.py) のサンプル画像命名規則（実出力で確認済み）:
#   {output_name}_e{epoch:06d}_{prompt_idx:02d}_{timestamp:%Y%m%d%H%M%S}[_{seed}][_{grid_idx}].png
# 例: krea2_lora_e000001_00_20260702104531_42_000.png
# prompt_idx はプロンプトファイル内の出現順 (0始まり) であり A/B のラベルとは無関係。
# grid_idx は末尾に付与される画像内インデックス（save_images_grid() 由来と推測、
# 単一画像出力時でも "_000" が付与されることを実出力で確認済み。未確証のソース由来）。
# seed 部分は GUI が固定値 (SAMPLE_FIXED_SEED / +1) を必ず付与するため、
# A/B の判別にはこちらを用いる。
_SAMPLE_FILENAME_RE = re.compile(
    r"_e(?P<epoch>\d+)_(?P<idx>\d+)_(?P<timestamp>\d{14})(?:_(?P<seed>\d+))?(?:_\d+)?$"
)


def _collect_images(image_dir: Path, seed: int | None = None) -> list[Path]:
    """image_dir から PNG ファイルを更新時刻降順で最大 10 件返す。

    seed を指定した場合、ファイル名末尾の seed サフィックスが一致する
    ものだけに絞り込む（A/B 判別フィルタ）。
    """
    if not image_dir.exists():
        return []
    files = sorted(
        image_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if seed is not None:
        files = [p for p in files if _extract_seed_from_filename(p) == seed]
    return files[:10]


def _extract_seed_from_filename(path: Path) -> int | None:
    """ファイル名から固定シード（サンプル A/B 判別キー）を抽出する。

    命名規則に一致しない、または seed サフィックスが存在しない場合は None を返す
    （呼び出し側でどの A/B にも属さないものとして除外される）。
    """
    m = _SAMPLE_FILENAME_RE.search(path.stem)
    if m is None or m.group("seed") is None:
        return None
    return int(m.group("seed"))


def _parse_epoch(path: Path) -> str:
    """ファイル名からエポック番号を抽出する（musubi-tuner 命名規則準拠）。"""
    m = _SAMPLE_FILENAME_RE.search(path.stem)
    return str(int(m.group("epoch"))) if m else "-"


def _resolve_sample_dir(s: TrainState) -> Path:
    """s.output_dir (tk.StringVar) から output_dir/sample を都度解決して返す。

    musubi-tuner (trainer_base.py) がサンプル画像の保存先として
    固定的に使用するディレクトリと同一パスを指す。output_dir は
    実行中に変更されうるため、呼び出しのたびに再評価する
    （構築時に一度だけ Path として固定してはならない）。
    """
    return Path(s.output_dir.get()) / "sample"
