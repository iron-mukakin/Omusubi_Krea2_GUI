"""app/tab_cache.py — [キャッシュ生成] タブ UI ビルダー。

parent への配置は grid のみ（labeled_frame は使わず ttk.LabelFrame を grid する）。
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .config import AppPaths
from .state import TrainState
from .tab_dataset import resolve_dataset_config, DATASET_MODE_GUI
from .widgets import attach_log_widget

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# ── 公開エントリポイント ─────────────────────────────────────────────────────

def build_cache_tab(parent: ttk.Frame, s: TrainState, root: tk.Tk) -> None:
    """キャッシュ生成タブの UI を parent に構築する（全 grid）。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(2, weight=1)

    # ── Step 1: Latent キャッシュ（row 0）────────────────────
    lf1 = ttk.LabelFrame(parent, text="Step 1: Latent キャッシュ生成（VAE エンコード）")
    lf1.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
    lf1.columnconfigure(1, weight=0)

    ttk.Label(
        lf1,
        text="VAE パス・dataset_config はモデルタブ / データセットタブの設定を使用します。",
        foreground="#64748B", justify=tk.LEFT,
    ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=(4, 4), pady=3)

    skip_latent = tk.BooleanVar(value=True)
    ttk.Checkbutton(lf1, text="skip_existing（既存キャッシュをスキップ）",
                    variable=skip_latent).grid(
        row=1, column=0, columnspan=3, sticky=tk.W, padx=(4, 4), pady=3)

    cache_log_widgets: list[tk.Text] = []

    ttk.Button(
        lf1, text="▶ Latent キャッシュを生成",
        command=lambda: _run_latent_cache(s, root, skip_latent.get(), cache_log_widgets),
    ).grid(row=2, column=0, sticky=tk.W, padx=(4, 4), pady=(4, 6))

    # ── Step 2: TE キャッシュ（row 1）────────────────────────
    lf2 = ttk.LabelFrame(parent,
                          text="Step 2: Text Encoder キャッシュ生成（Qwen3-VL エンコード）")
    lf2.grid(row=1, column=0, sticky=tk.EW, pady=(0, 6))
    lf2.columnconfigure(1, weight=0)
    lf2.columnconfigure(3, weight=0)

    ttk.Label(
        lf2,
        text="Text Encoder パスはモデルタブの「Text Encoder」を使用します（必須）。",
        foreground="#64748B", justify=tk.LEFT,
    ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    ttk.Label(
        lf2,
        text=(
            "⚠️  ComfyUI の fp8 量子化版（comfy_quant キー含む）は使用不可です。\n"
            "    bf16 版（非量子化）の Qwen3-VL-4B を使用してください。\n"
            "    例: Qwen/Qwen3-VL-4B-Instruct の bf16 safetensors\n"
            "    または ComfyUI の fp8_scaled.safetensors ではなく bf16 版を指定してください。"
        ),
        foreground="#EF4444", justify=tk.LEFT,
    ).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=(0, 4))

    te_batch    = tk.IntVar(value=1)
    skip_te     = tk.BooleanVar(value=True)
    te_dtype    = tk.StringVar(value="bfloat16")

    ttk.Label(lf2, text="batch_size", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=16, textvariable=te_batch, width=6).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="text_encoder_dtype", width=20, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf2, textvariable=te_dtype,
                 values=["bfloat16", "float16", "float32"],
                 state="readonly", width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Checkbutton(lf2, text="skip_existing（既存キャッシュをスキップ）",
                    variable=skip_te).grid(
        row=3, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    ttk.Button(
        lf2, text="▶ Text Encoder キャッシュを生成",
        command=lambda: _run_te_cache(
            s, root, te_batch.get(), skip_te.get(),
            te_dtype.get(), cache_log_widgets),
    ).grid(row=4, column=0, sticky=tk.W, padx=(4, 4), pady=(4, 6))

    # ── ログ（row 2）─────────────────────────────────────────
    log_lf = ttk.LabelFrame(parent, text="キャッシュ生成ログ")
    log_lf.grid(row=2, column=0, sticky=tk.NSEW, pady=(0, 4))
    log_lf.columnconfigure(0, weight=1)
    log_lf.rowconfigure(0, weight=1)
    attach_log_widget(log_lf, cache_log_widgets, height=12)

    # ── 注記（row 3）─────────────────────────────────────────
    ttk.Label(
        parent,
        text=(
            "【手順】Step 1（Latent）→ Step 2（TE）の順で実行してください。\n"
            "両方完了後に [▶ 実行] タブで学習を開始できます。"
        ),
        foreground="#1D4ED8", justify=tk.LEFT,
    ).grid(row=3, column=0, sticky=tk.W, padx=6, pady=(4, 0))


# ── Latent キャッシュ実行 ────────────────────────────────────────────────────

def _run_latent_cache(
    s: TrainState,
    root: tk.Tk,
    skip_existing: bool,
    log_widgets: list[tk.Text],
) -> None:
    err = _check_common(s)
    if err:
        messagebox.showerror("エラー", err)
        return
    if not s.vae_path.get().strip():
        messagebox.showerror("エラー",
                             "VAE パスが未指定です（モデルタブで設定してください）。")
        return
    try:
        dataset_config = resolve_dataset_config(s)
    except ValueError as e:
        messagebox.showerror("エラー", str(e))
        return

    cmd = [
        str(s.paths.musubi_venv_py),
        str(s.paths.krea2_cache_latents_script),
        "--dataset_config", dataset_config,
        "--vae", s.vae_path.get().strip(),
    ]
    if skip_existing:
        cmd.append("--skip_existing")

    _log(log_widgets, f"[Latent] {' '.join(cmd)}", root)
    _launch(cmd, s, log_widgets, root, "Latent キャッシュ")


# ── TE キャッシュ実行 ────────────────────────────────────────────────────────

def _run_te_cache(
    s: TrainState,
    root: tk.Tk,
    batch_size: int,
    skip_existing: bool,
    te_dtype: str,
    log_widgets: list[tk.Text],
) -> None:
    err = _check_common(s)
    if err:
        messagebox.showerror("エラー", err)
        return
    if not s.text_encoder_path.get().strip():
        messagebox.showerror("エラー",
                             "Text Encoder パスが未指定です（モデルタブで設定してください）。")
        return
    try:
        dataset_config = resolve_dataset_config(s)
    except ValueError as e:
        messagebox.showerror("エラー", str(e))
        return

    cmd = [
        str(s.paths.musubi_venv_py),
        str(s.paths.krea2_cache_te_script),
        "--dataset_config",     dataset_config,
        "--text_encoder",       s.text_encoder_path.get().strip(),
        "--batch_size",         str(batch_size),
        "--text_encoder_dtype", te_dtype,
    ]
    if skip_existing:
        cmd.append("--skip_existing")

    _log(log_widgets, f"[TE] {' '.join(cmd)}", root)
    _launch(cmd, s, log_widgets, root, "TE キャッシュ")


# ── 内部ヘルパー ─────────────────────────────────────────────────────────────

def _check_common(s: TrainState) -> str | None:
    err = s.paths.validate_musubi()
    if err:
        return err
    if s.dataset_mode.get() == DATASET_MODE_GUI:
        if not any(ev.image_dir.get().strip() for ev in s.dataset_entries):
            return "画像ディレクトリが未設定です（データセットタブで設定してください）。"
    else:
        if not s.dataset_config_path.get().strip():
            return "dataset_config (toml) が未指定です。"
    return None


def _launch(
    cmd: list[str],
    s: TrainState,
    log_widgets: list[tk.Text],
    root: tk.Tk,
    label: str,
) -> None:
    src_path = str(s.paths.musubi_src)

    def _worker() -> None:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(s.paths.musubi),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
            )
            for raw in proc.stdout:
                _log(log_widgets, _ANSI_RE.sub("", raw.rstrip()), root)
            proc.wait()
            _log(log_widgets, f"[{label}] 完了 (rc={proc.returncode})", root)
        except Exception as exc:
            _log(log_widgets, f"[ERROR] {exc}", root)

    threading.Thread(target=_worker, daemon=True).start()


def _log(log_widgets: list[tk.Text], msg: str, root: tk.Tk | None = None) -> None:
    def _append() -> None:
        for w in log_widgets:
            w.configure(state=tk.NORMAL)
            w.insert(tk.END, msg + "\n")
            w.see(tk.END)
            w.configure(state=tk.DISABLED)
    if root is not None:
        root.after(0, _append)
    else:
        _append()
