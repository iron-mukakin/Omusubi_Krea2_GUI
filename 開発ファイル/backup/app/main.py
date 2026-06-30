"""app/main.py — アプリケーションエントリポイント。

起動方法:
    python -m app.main          # GUIルートから（推奨）
    python app/main.py          # 直接実行（sys.path を自動補正）
"""
from __future__ import annotations

import sys
import os

# ── 直接実行時の sys.path 補正 ──────────────────────────────────────────────
# `python app/main.py` で起動された場合、app/ の親ディレクトリが
# sys.path に入っていないため相対 import が失敗する。
# `python -m app.main` では不要だが、両方の起動方式に対応するため補正する。
if __name__ == "__main__" and __package__ is None:
    _gui_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _gui_root not in sys.path:
        sys.path.insert(0, _gui_root)
    # パッケージとして再 import して相対 import を有効にする
    import importlib
    import runpy
    runpy.run_module("app.main", run_name="__main__", alter_sys=True)
    sys.exit(0)

import tkinter as tk
from tkinter import ttk

from .config import AppPaths
from .tab_lora import build_lora_tab

APP_TITLE  = "Musubi LoRA GUI"
WIN_SIZE   = "1200x820"
MIN_WIDTH  = 900
MIN_HEIGHT = 600


def _setup_styles(root: tk.Tk) -> None:
    """アプリ全体の ttk スタイルを設定する。"""
    style = ttk.Style(root)
    for preferred in ("vista", "clam", "alt", "default"):
        if preferred in style.theme_names():
            style.theme_use(preferred)
            break


def _build_status_bar(root: tk.Tk) -> ttk.Label:
    """ウィンドウ下部にステータスバーを配置して Label を返す。"""
    bar = ttk.Label(
        root, text="準備完了", anchor=tk.W,
        relief=tk.SUNKEN, padding=(4, 2),
    )
    bar.pack(side=tk.BOTTOM, fill=tk.X)
    return bar


def _make_log_fn(status_bar: ttk.Label, log_text: tk.Text):
    """グローバルログ出力関数を生成して返す。"""
    import datetime

    def _log(msg: str) -> None:
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        log_text.configure(state=tk.NORMAL)
        log_text.insert(tk.END, full + "\n")
        log_text.see(tk.END)
        log_text.configure(state=tk.DISABLED)
        status_bar.configure(text=msg[:120])

    return _log


def main() -> None:
    """アプリケーション本体。"""
    paths = AppPaths.from_root()
    paths.ensure_dirs()

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry(WIN_SIZE)
    root.minsize(MIN_WIDTH, MIN_HEIGHT)
    _setup_styles(root)

    # ステータスバー（最下段）
    status_bar = _build_status_bar(root)

    # グローバルログ領域
    log_frame = ttk.LabelFrame(root, text="ログ", padding=2)
    log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0, 4))
    log_text = tk.Text(
        log_frame, height=4, wrap=tk.WORD,
        font=("TkFixedFont", 8), state=tk.DISABLED,
    )
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.X)

    log_fn = _make_log_fn(status_bar, log_text)

    # グローバルノートブック（機能タブ）
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    lora_frame = ttk.Frame(nb)
    nb.add(lora_frame, text="Krea2 LoRA 学習")
    build_lora_tab(lora_frame, paths, root, log_fn)

    # 将来のタブ追加例:
    # merge_frame = ttk.Frame(nb)
    # nb.add(merge_frame, text="マージ")
    # build_merge_tab(merge_frame, paths, root, log_fn)

    log_fn(f"起動完了。musubi-tuner: {paths.musubi}")

    err = paths.validate_musubi()
    if err:
        log_fn(f"[WARN] {err}")

    root.mainloop()


if __name__ == "__main__":
    main()
