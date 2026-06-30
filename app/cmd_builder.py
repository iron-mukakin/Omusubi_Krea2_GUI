"""app/cmd_builder.py — Krea2 LoRA 学習コマンド生成。

accelerate launch → _krea2_wrapper.py → krea2_train_network.py
の3段呼び出しリストを生成し、wrapper ファイルを書き出す。
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from .config import AppPaths
from .state import TrainState, DATASET_MODE_TOML, SAMPLE_FIXED_SEED, LAYER_GROUP_DEFS

# LoRA ターゲット別の network_args（exclude_patterns）
_LORA_EXCLUDE_ATTENTION_ONLY: str = (
    "exclude_patterns=["
    "'.*\\.mlp\\..*','first','last\\.linear',"
    "'tmlp\\..*','txtmlp\\..*','tproj\\.1','txtfusion\\..*'"
    "]"
)


def write_wrapper_script(paths: AppPaths) -> Path:
    """Windows spawn 安全な launcher スクリプトを生成して返す。

    exec() 廃止理由:
      Windows multiprocessing spawn モードでは DataLoader worker が
      メインスクリプトを再 import する。exec() はトップレベルで実行されるため
      if __name__ == '__main__' ガードが機能せず RuntimeError が発生する。

    代替設計:
      runpy.run_module() + if __name__ == '__main__' + freeze_support() で保護。
      spawn worker が launcher を再 import しても __name__ != '__main__' のため
      run_module は呼ばれない。絶対パスは launcher 内に埋め込まず
      sys.path への追加のみ行う（パスは相対参照で解決）。
    """
    import os as _os
    src_path    = paths.musubi_src
    musubi_root = paths.musubi
    launcher_path = paths.app / "_krea2_launcher.py"

    # バックスラッシュを / に正規化（toml / Python 文字列リテラル両対応）
    src_str  = _os.fspath(src_path).replace("\\", "/")
    root_str = _os.fspath(musubi_root).replace("\\", "/")

    lines = [
        "# _krea2_launcher.py — musubi-tuner Krea2 学習ランチャー",
        "# GUIアプリが自動生成します。手動編集不要。",
        "# Windows multiprocessing spawn 対応: exec() を使用しない。",
        "import sys",
        "import os",
        "import runpy",
        "from multiprocessing import freeze_support",
        "",
        f"_SRC  = '{src_str}'",
        f"_ROOT = '{root_str}'",
        "",
        "if _SRC not in sys.path:",
        "    sys.path.insert(0, _SRC)",
        "os.chdir(_ROOT)",
        "",
        "if __name__ == '__main__':",
        "    freeze_support()",
        "    runpy.run_module(",
        "        'musubi_tuner.krea2_train_network',",
        "        run_name='__main__',",
        "        alter_sys=True,",
        "    )",
    ]
    launcher_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return launcher_path


def validate_inputs(s: TrainState) -> str | None:
    """必須入力のバリデーション。エラーメッセージを返す（正常時 None）。"""
    if not s.dit_path.get():
        return "DiT (RAW) パスが未指定です。"
    if not Path(s.dit_path.get()).exists():
        return f"DiT ファイルが見つかりません: {s.dit_path.get()}"
    if not s.vae_path.get():
        return "VAE パスが未指定です。"
    if not Path(s.vae_path.get()).exists():
        return f"VAE ファイルが見つかりません: {s.vae_path.get()}"

    # データセット検証（モード別）
    if s.dataset_mode.get() == DATASET_MODE_TOML:
        p = s.dataset_config_path.get().strip()
        if not p:
            return "toml ファイルパスが未指定です。"
        if not Path(p).exists():
            return f"dataset_config が見つかりません: {p}"
    else:
        valid_entries = [ev for ev in s.dataset_entries
                         if ev.image_dir.get().strip()]
        if not valid_entries:
            return "画像ディレクトリが1件も設定されていません。"
        for i, ev in enumerate(valid_entries):
            d = ev.image_dir.get().strip()
            if not Path(d).exists():
                return f"エントリ {i + 1}: 画像ディレクトリが見つかりません: {d}"

    if s.sample_enabled.get() and not s.sample_prompt.get().strip():
        return "サンプル A が有効ですがプロンプトが空です。"
    if s.sample_b_enabled.get() and not s.sample_b_prompt.get().strip():
        return "サンプル B が有効ですがプロンプトが空です。"
    # Krea2 制約: fp8_base 単体は不可。base+scaled の両方が必要
    if s.fp8_base.get() and not s.fp8_scaled.get():
        return (
            "Krea2 の fp8 は --fp8_base と --fp8_scaled の両方が必要です。\n"
            "[詳細・最適化] タブの fp8 チェックを確認してください。"
        )
    if s.fp8_scaled.get() and not s.fp8_base.get():
        return "--fp8_scaled を使用するには --fp8_base も有効にしてください。"
    # Krea2 制約: turbo_dit と blocks_to_swap は併用不可
    if s.turbo_dit_path.get() and s.blocks_to_swap.get() > 0:
        return "--turbo_dit と --blocks_to_swap は同時に使用できません。"
    # turbo_dit_cache は turbo_dit が必要
    if s.turbo_dit_cache.get() and not s.turbo_dit_path.get():
        return "--turbo_dit_cache には --turbo_dit の指定が必要です。"
    return None


def build_train_command(s: TrainState, paths: AppPaths) -> list[str]:
    """accelerate launch コマンドリストを生成して返す。

    launcher スクリプトは write_wrapper_script() が生成する。
    launcher は if __name__ == '__main__' + freeze_support() で保護されており
    Windows spawn モードの DataLoader worker による再実行でも安全。
    PYTHONPATH は launcher 内の sys.path 操作で設定するため env への注入不要。
    """
    from .tab_dataset import resolve_dataset_config
    dataset_config_path = resolve_dataset_config(s)

    launcher = write_wrapper_script(paths)
    python   = str(paths.musubi_venv_py)

    cmd: list[str] = [
        python, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", s.mixed_precision.get(),
        str(launcher),
        # ── 必須引数 ──────────────────────────────────────────────
        "--dit",             s.dit_path.get(),
        "--vae",             s.vae_path.get(),
        "--dataset_config",  dataset_config_path,
        "--network_module",  "networks.lora_krea2",
        "--network_dim",     str(s.network_dim.get()),
        "--network_alpha",   str(s.network_alpha.get()),
        "--output_dir",      s.output_dir.get(),
        "--output_name",     s.output_name.get(),
        # ── 学習設定 ──────────────────────────────────────────────
        "--learning_rate",               s.learning_rate.get(),
        "--lr_scheduler",                s.lr_scheduler.get(),
        "--lr_warmup_steps",             str(s.lr_warmup_steps.get()),
        "--optimizer_type",              s.optimizer_type.get(),
        "--max_train_epochs",            str(s.max_train_epochs.get()),
        "--save_every_n_epochs",         str(s.save_every_n_epochs.get()),
        "--mixed_precision",             s.mixed_precision.get(),
        "--save_precision",              s.save_precision.get(),
        "--gradient_accumulation_steps", str(s.grad_accum.get()),
        "--max_grad_norm",               str(s.max_grad_norm.get()),
        # NOTE: batch_size / num_workers は dataset_config.toml [general] に記述済み。
        #       krea2_train_network.py は --train_batch_size / --max_data_loader_n_workers
        #       を受け付けないため CLI には含めない。
        # ── Krea2 詳細 ────────────────────────────────────────────
        "--timestep_sampling",   s.timestep_sampling.get(),
        "--discrete_flow_shift", str(s.discrete_flow_shift.get()),
        "--weighting_scheme",    s.weighting_scheme.get(),
    ]

    # ── オプション文字列引数 ────────────────────────────────────
    _append_optional_str(cmd, "--seed",            s.seed.get())
    _append_optional_str(cmd, "--text_encoder",    s.text_encoder_path.get())
    _append_optional_str(cmd, "--turbo_dit",       s.turbo_dit_path.get())
    _append_optional_str(cmd, "--network_weights", s.network_weights.get())

    if s.optimizer_args.get().strip():
        cmd += ["--optimizer_args"] + s.optimizer_args.get().split()

    # ── LoRA ターゲット (network_args) ──────────────────────────
    target = s.lora_target.get()
    if target == "attention_only":
        cmd += ["--network_args", _LORA_EXCLUDE_ATTENTION_ONLY]
    elif target == "custom":
        if s.layer_input_mode.get() == "group":
            # 層グループ選択モード: OFF グループから exclude_patterns を生成
            args_str = _build_network_args_from_groups(s)
            if args_str:
                cmd += ["--network_args", args_str]
            # args_str 空 = 全 ON → --network_args なし（全層対象）
        else:
            # テキスト直接入力モード: network_args をそのまま使用
            if s.network_args.get().strip():
                cmd += ["--network_args", s.network_args.get().strip()]

    # ── Attention モード ─────────────────────────────────────────
    attn_flag = {
        "sdpa":     "--sdpa",
        "flash":    "--flash_attn",
        "sageattn": "--sage_attn",
        "xformers": "--xformers",
    }.get(s.attn_mode.get())
    if attn_flag:
        cmd.append(attn_flag)

    # ── Bool フラグ ──────────────────────────────────────────────
    bool_flags: list[tuple[tk.BooleanVar, str]] = [
        (s.gradient_checkpointing,     "--gradient_checkpointing"),
        (s.persistent_workers,         "--persistent_data_loader_workers"),
        (s.split_attn,                 "--split_attn"),
        (s.fp8_base,                   "--fp8_base"),
        (s.fp8_scaled,                 "--fp8_scaled"),
        (s.block_swap_h2d_only,        "--block_swap_h2d_only"),
        (s.use_pinned_memory,          "--use_pinned_memory_for_block_swap"),
        (s.cpu_offload_checkpointing,  "--cpu_offload_checkpointing"),
        (s.compile,                    "--compile"),
        (s.turbo_dit_cache,            "--turbo_dit_cache"),
    ]
    for var, flag in bool_flags:
        if var.get():
            cmd.append(flag)

    # Krea2 fp8 安全ガード: fp8_base ON なら fp8_scaled も必ず送信
    if s.fp8_base.get() and "--fp8_scaled" not in cmd:
        cmd.append("--fp8_scaled")

    # ── Block Swap ───────────────────────────────────────────────
    bts = s.blocks_to_swap.get()
    if bts > 0:
        cmd += ["--blocks_to_swap", str(bts),
                "--block_swap_ring_size", str(s.block_swap_ring_size.get())]

    # ── サンプル生成 ─────────────────────────────────────────────
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        sample_file = _write_sample_prompt_file(s)
        cmd += [
            "--sample_every_n_epochs",
            s.sample_every_n_epochs.get().strip() or "1",
            "--sample_prompts", str(sample_file),
        ]

    return cmd


# ── 内部ヘルパー ─────────────────────────────────────────────────────────────

def _build_network_args_from_groups(s: TrainState) -> str:
    """層グループ選択の状態から network_args 文字列を生成して返す。

    Returns:
        exclude_patterns=[...] 形式の文字列。
        全グループ ON の場合は空文字（--network_args 不要）。
    """
    off_keys = [key for key, var in s.layer_groups.items() if not var.get()]
    if not off_keys:
        return ""

    pat_map  = {key: pat for key, _label, pat in LAYER_GROUP_DEFS}
    patterns = [pat_map[k] for k in off_keys if k in pat_map]
    if not patterns:
        return ""

    quoted = ", ".join("'" + p + "'" for p in patterns)
    return "exclude_patterns=[" + quoted + "]"


def _append_optional_str(cmd: list[str], flag: str, value: str) -> None:
    """value が空でなければ flag と value をコマンドリストに追加する。"""
    if value.strip():
        cmd += [flag, value.strip()]


def _write_sample_prompt_file(s: TrainState) -> Path:
    """サンプルプロンプトファイルを書き出してパスを返す。"""
    w     = s.sample_width.get()
    h     = s.sample_height.get()
    steps = s.sample_steps.get()
    scale = s.sample_guidance_scale.get()

    def _line(prompt: str, neg: str, seed: int) -> str:
        base = f"{prompt} --w {w} --h {h} --s {steps} --l {scale:g} --d {seed}"
        return base + (f" --n {neg}" if neg else "")

    lines: list[str] = []
    if s.sample_enabled.get() and s.sample_prompt.get().strip():
        lines.append(_line(
            s.sample_prompt.get().strip(),
            s.sample_negative_prompt.get().strip(),
            SAMPLE_FIXED_SEED,
        ))
    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():
        lines.append(_line(
            s.sample_b_prompt.get().strip(),
            s.sample_b_negative_prompt.get().strip(),
            SAMPLE_FIXED_SEED + 1,
        ))

    out_dir     = s.paths.log / "sample_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = out_dir / "_sample_prompt.txt"
    prompt_file.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return prompt_file
