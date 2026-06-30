"""app/tab_lora.py — Krea2 LoRA 学習メインタブ オーケストレーター。

サブタブを Notebook に組み立てる。実行パネル（コマンドプレビュー・開始・停止・
ログ）は専用の「実行」タブに集約し、設定タブはスクロールなしで完結させる。
将来のマージタブ等はグローバル Notebook（main.py）側に並べる形で拡張する。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .config import AppPaths
from .state import TrainState
from .tab_cache    import build_cache_tab
from .tab_model    import build_model_tab
from .tab_dataset  import build_dataset_tab
from .tab_network  import build_network_tab
from .tab_train    import build_train_tab
from .tab_advanced import build_advanced_tab
from .tab_sample   import build_sample_tab
from .tab_monitor  import build_monitor_tab
from .tab_preset   import build_preset_tab
from .run_panel    import build_run_panel


def build_lora_tab(
    parent: ttk.Frame,
    paths: AppPaths,
    root: tk.Tk,
    log_fn,
) -> TrainState:
    """LoRA 学習タブ全体を parent に構築し、TrainState を返す。

    Parameters
    ----------
    parent:  このタブが配置される Frame
    paths:   AppPaths インスタンス
    root:    Tk ルートウィンドウ（after() 登録用）
    log_fn:  グローバルログ出力関数
    """
    state = TrainState(paths=paths, log_fn=log_fn)

    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.BOTH, expand=True)

    # ── 設定サブタブ ─────────────────────────────────────────
    setting_specs: list[tuple[str, object]] = [
        ("モデル",       build_model_tab),
        ("データセット", build_dataset_tab),
        ("キャッシュ生成", build_cache_tab),
        ("ネットワーク", build_network_tab),
        ("学習設定",     build_train_tab),
        ("詳細・最適化", build_advanced_tab),
        ("サンプル生成", build_sample_tab),
        ("モニター",     build_monitor_tab),
        ("プリセット",   build_preset_tab),
    ]
    for label, builder_fn in setting_specs:
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text=label)
        if builder_fn is build_cache_tab:
            builder_fn(tab, state, root)
        else:
            builder_fn(tab, state)

    # ── 実行タブ（コマンドプレビュー・開始・停止・ログ）─────
    run_tab = ttk.Frame(nb, padding=8)
    nb.add(run_tab, text="▶ 実行")
    build_run_panel(run_tab, state, root)

    return state
