"""app/tab_advanced.py — [詳細・メモリ最適化] タブ UI ビルダー。

Krea2 固有の排他制約をウィジェット連動で防止する:

  [A] turbo_dit × blocks_to_swap
      turbo_dit パスが入力されたら blocks_to_swap を 0 に強制リセットし
      Spinbox をロック（disabled）。解除は turbo_dit を空にする。

  [B] fp8_scaled → fp8_base 強制 ON（逆に base OFF → scaled も OFF）

  [C] turbo_dit_cache は turbo_dit が必要
      turbo_dit 空のまま cache にチェックを入れようとしたら自動解除 + ツールチップ

  [D] block_swap_* オプションは blocks_to_swap=0 のときグレーアウト
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .state import TrainState, TIMESTEP_SAMPLINGS, WEIGHTING_SCHEMES, ATTN_MODES
from .widgets import labeled_frame


def build_advanced_tab(parent: ttk.Frame, s: TrainState) -> None:
    """詳細・メモリ最適化タブの UI を parent に構築する。"""

    # ── タイムステップ / 重み付け ─────────────────────────────
    lf = labeled_frame(parent, "タイムステップ / 重み付け")
    lf.columnconfigure(1, weight=0)
    lf.columnconfigure(3, weight=0)

    ttk.Label(lf, text="timestep_sampling", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.timestep_sampling,
                 values=list(TIMESTEP_SAMPLINGS), state="readonly", width=16).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)

    ttk.Label(lf, text="discrete_flow_shift", width=18, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf, textvariable=s.discrete_flow_shift, width=10).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="weighting_scheme", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.weighting_scheme,
                 values=list(WEIGHTING_SCHEMES), state="readonly", width=16).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 4), pady=3)
    ttk.Label(lf, text="shift=2.5 が 1024px 推奨 / krea2_shift で自動調整",
              foreground="#64748B").grid(
        row=1, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    # ── Attention モード ──────────────────────────────────────
    lf2 = labeled_frame(parent, "Attention モード")
    lf2.columnconfigure(1, weight=0)

    ttk.Label(lf2, text="attn_mode", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.attn_mode,
                 values=list(ATTN_MODES), state="readonly", width=14).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Checkbutton(lf2, text="split_attn（sdpa 以外で推奨）",
                    variable=s.split_attn).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 4), pady=3)

    # ── Block Swap ───────────────────────────────────────────
    lf3 = labeled_frame(parent, "Block Swap（VRAM 節約）  ※ turbo_dit 使用時は無効")
    lf3.columnconfigure(1, weight=0)
    lf3.columnconfigure(3, weight=0)

    ttk.Label(lf3, text="blocks_to_swap (0-26)", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    bts_spin = ttk.Spinbox(lf3, from_=0, to=26,
                            textvariable=s.blocks_to_swap, width=8)
    bts_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)

    ttk.Label(lf3, text="block_swap_ring_size", width=18, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ring_spin = ttk.Spinbox(lf3, from_=1, to=8,
                             textvariable=s.block_swap_ring_size, width=6)
    ring_spin.grid(row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    chk_row = ttk.Frame(lf3)
    chk_row.grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)
    h2d_chk = ttk.Checkbutton(chk_row, text="block_swap_h2d_only",
                               variable=s.block_swap_h2d_only)
    h2d_chk.pack(side=tk.LEFT, padx=(0, 12))
    pin_chk = ttk.Checkbutton(chk_row, text="use_pinned_memory_for_block_swap",
                               variable=s.use_pinned_memory)
    pin_chk.pack(side=tk.LEFT)

    # block swap 関連ウィジェット一覧（turbo_dit 時に一括 disable）
    bts_widgets = [bts_spin, ring_spin, h2d_chk, pin_chk]

    # ── fp8 ──────────────────────────────────────────────────
    # Krea2 は --fp8_base + --fp8_scaled のセットのみ有効。
    # UI では単一チェックで両方を同時に ON/OFF する。
    lf4 = labeled_frame(parent, "fp8 量子化（DiT 重みメモリ半減）")

    fp8_row = ttk.Frame(lf4)
    fp8_row.pack(anchor=tk.W, padx=4, pady=3)

    ttk.Checkbutton(
        fp8_row,
        text="fp8 を有効にする（fp8_base + fp8_scaled を同時適用）",
        variable=s.fp8_base,
        command=lambda: _on_fp8_toggle(s),
    ).pack(side=tk.LEFT)

    ttk.Label(lf4,
              text="有効にすると --fp8_base --fp8_scaled の両方が渡されます。",
              foreground="#64748B").pack(anchor=tk.W, padx=4, pady=(0, 3))

    # ── Turbo DiT ────────────────────────────────────────────
    lf5 = labeled_frame(parent, "Turbo DiT  ※ block_swap と排他")
    lf5.columnconfigure(1, weight=1)

    ttk.Label(lf5, text="turbo_dit パス", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    turbo_entry = ttk.Entry(lf5, textvariable=s.turbo_dit_path)
    turbo_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 2), pady=3)
    ttk.Button(lf5, text="Browse", width=7,
               command=lambda: _pick_turbo(s)).grid(
        row=0, column=2, padx=(0, 4), pady=3)

    turbo_cache_cb = ttk.Checkbutton(
        lf5, text="turbo_dit_cache（M1/CPU 常駐）",
        variable=s.turbo_dit_cache,
        command=lambda: _on_turbo_cache(s),
    )
    turbo_cache_cb.grid(row=1, column=0, columnspan=3,
                        sticky=tk.W, padx=(4, 4), pady=3)

    _turbo_hint = ttk.Label(lf5, text="", foreground="#EF4444")
    _turbo_hint.grid(row=2, column=0, columnspan=3,
                     sticky=tk.W, padx=(4, 4), pady=(0, 3))

    # turbo_dit パスの変更を監視して block swap を連動制御
    s.turbo_dit_path.trace_add(
        "write",
        lambda *_: _on_turbo_path_change(s, bts_widgets, _turbo_hint),
    )

    # ── その他の最適化 ────────────────────────────────────────
    lf6 = labeled_frame(parent, "その他の最適化")

    other_row = ttk.Frame(lf6)
    other_row.pack(anchor=tk.W, padx=4, pady=3)
    ttk.Checkbutton(other_row, text="cpu_offload_checkpointing",
                    variable=s.cpu_offload_checkpointing).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Checkbutton(other_row, text="compile (torch.compile)",
                    variable=s.compile).pack(side=tk.LEFT)

    # 初期状態を適用
    _on_turbo_path_change(s, bts_widgets, _turbo_hint)


# ── 排他制御コールバック ─────────────────────────────────────────────────────

def _on_turbo_path_change(
    s: TrainState,
    bts_widgets: list[tk.Widget],
    hint_lbl: ttk.Label,
) -> None:
    """turbo_dit パスが入力されたら block_swap を無効化する。"""
    has_turbo = bool(s.turbo_dit_path.get().strip())
    new_state = tk.DISABLED if has_turbo else "!disabled"

    for w in bts_widgets:
        try:
            w.state([new_state])        # ttk ウィジェット
        except AttributeError:
            w.configure(state=tk.DISABLED if has_turbo else tk.NORMAL)

    if has_turbo:
        # blocks_to_swap を 0 に強制リセット
        if s.blocks_to_swap.get() > 0:
            s.blocks_to_swap.set(0)
        hint_lbl.configure(
            text="turbo_dit 使用中: blocks_to_swap は使用できません。")
    else:
        hint_lbl.configure(text="")


def _on_fp8_toggle(s: TrainState) -> None:
    """fp8 チェックボックスの ON/OFF に合わせて fp8_scaled を同期する。

    Krea2 は --fp8_base 単体不可（ValueError）のため、
    fp8_base と fp8_scaled を常に同じ値に保つ。
    """
    s.fp8_scaled.set(s.fp8_base.get())


def _on_turbo_cache(s: TrainState) -> None:
    """turbo_dit_cache は turbo_dit パスが必要。空なら自動解除。"""
    if s.turbo_dit_cache.get() and not s.turbo_dit_path.get().strip():
        s.turbo_dit_cache.set(False)


def _pick_turbo(s: TrainState) -> None:
    from tkinter import filedialog
    path = filedialog.askopenfilename(
        filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
    if path:
        s.turbo_dit_path.set(path)
