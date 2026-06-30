"""app/preset_manager.py — プリセットの収集・適用・保存・読込。

TrainState の tk.Variable 値を JSON に変換（収集）し、
JSON から TrainState へ復元（適用）する純粋変換層。
GUI（tkinter）への依存は tk.Variable のみ。
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

from .state import TrainState, DATASET_MODE_GUI
from .dataset_config import DatasetEntry


def collect_preset(s: TrainState) -> dict:
    """TrainState の全変数値を dict に変換して返す。"""
    # データセットエントリをシリアライズ
    entries_data = [
        {
            "image_dir":         ev.image_dir.get(),
            "cache_directory":   ev.cache_directory.get(),
            "num_repeats":       int(ev.num_repeats.get()),
            "caption_extension": ev.caption_extension.get(),
            "resolution":        int(ev.resolution.get()),
            "enable_bucket":     bool(ev.enable_bucket.get()),
            "bucket_no_upscale": bool(ev.bucket_no_upscale.get()),
            "batch_size":        int(ev.batch_size.get()),
        }
        for ev in s.dataset_entries
    ]

    return {
        # モデル
        "dit_path":           s.dit_path.get(),
        "vae_path":           s.vae_path.get(),
        "text_encoder_path":  s.text_encoder_path.get(),
        "turbo_dit_path":     s.turbo_dit_path.get(),
        "output_dir":         s.output_dir.get(),
        "output_name":        s.output_name.get(),
        "save_precision":     s.save_precision.get(),
        # データセット（モード + [general] + エントリ）
        "dataset_mode":                  s.dataset_mode.get(),
        "dataset_config_path":           s.dataset_config_path.get(),
        "general_resolution":            int(s.general_resolution.get()),
        "general_caption_extension":     s.general_caption_extension.get(),
        "general_enable_bucket":         bool(s.general_enable_bucket.get()),
        "general_bucket_no_upscale":     bool(s.general_bucket_no_upscale.get()),
        "batch_size":                    int(s.batch_size.get()),
        "num_workers":                   int(s.num_workers.get()),
        "persistent_workers":            bool(s.persistent_workers.get()),
        "dataset_entries":               entries_data,
        # ネットワーク
        "network_dim":        int(s.network_dim.get()),
        "network_alpha":      float(s.network_alpha.get()),
        "lora_target":        s.lora_target.get(),
        "network_args":       s.network_args.get(),
        "network_weights":    s.network_weights.get(),
        # 層グループ選択
        "layer_input_mode":   s.layer_input_mode.get(),
        "layer_groups":       {
            key: var.get() for key, var in s.layer_groups.items()
        },
        # 学習設定
        "learning_rate":      s.learning_rate.get(),
        "lr_scheduler":       s.lr_scheduler.get(),
        "lr_warmup_steps":    int(s.lr_warmup_steps.get()),
        "optimizer_type":     s.optimizer_type.get(),
        "optimizer_args":     s.optimizer_args.get(),
        "max_train_epochs":   int(s.max_train_epochs.get()),
        "save_every_n_epochs": int(s.save_every_n_epochs.get()),
        "seed":               s.seed.get(),
        "mixed_precision":    s.mixed_precision.get(),
        "grad_accum":         int(s.grad_accum.get()),
        "max_grad_norm":      float(s.max_grad_norm.get()),
        "gradient_checkpointing": bool(s.gradient_checkpointing.get()),
        # Krea2 詳細
        "timestep_sampling":  s.timestep_sampling.get(),
        "discrete_flow_shift": float(s.discrete_flow_shift.get()),
        "weighting_scheme":   s.weighting_scheme.get(),
        # Attention
        "attn_mode":          s.attn_mode.get(),
        "split_attn":         bool(s.split_attn.get()),
        # メモリ最適化
        "blocks_to_swap":     int(s.blocks_to_swap.get()),
        "fp8_base":           bool(s.fp8_base.get()),
        "fp8_scaled":         bool(s.fp8_scaled.get()),
        "block_swap_h2d_only": bool(s.block_swap_h2d_only.get()),
        "use_pinned_memory":  bool(s.use_pinned_memory.get()),
        "block_swap_ring_size": int(s.block_swap_ring_size.get()),
        "cpu_offload_checkpointing": bool(s.cpu_offload_checkpointing.get()),
        "compile":            bool(s.compile.get()),
        "turbo_dit_cache":    bool(s.turbo_dit_cache.get()),
        # サンプル生成
        "sample_enabled":     bool(s.sample_enabled.get()),
        "sample_every_n_epochs": s.sample_every_n_epochs.get(),
        "sample_prompt":      s.sample_prompt.get(),
        "sample_negative_prompt": s.sample_negative_prompt.get(),
        "sample_b_enabled":   bool(s.sample_b_enabled.get()),
        "sample_b_prompt":    s.sample_b_prompt.get(),
        "sample_b_negative_prompt": s.sample_b_negative_prompt.get(),
        "sample_width":       int(s.sample_width.get()),
        "sample_height":      int(s.sample_height.get()),
        "sample_steps":       int(s.sample_steps.get()),
        "sample_guidance_scale": float(s.sample_guidance_scale.get()),
    }


def apply_preset(s: TrainState, data: dict) -> None:
    """dict の値を TrainState の各 tk.Variable に反映する。"""

    def _set(var: tk.Variable, key: str, default=None) -> None:
        if key in data:
            try:
                var.set(data[key])
            except (tk.TclError, ValueError):
                if default is not None:
                    var.set(default)

    _set(s.dit_path,           "dit_path",           "")
    _set(s.vae_path,           "vae_path",           "")
    _set(s.text_encoder_path,  "text_encoder_path",  "")
    _set(s.turbo_dit_path,     "turbo_dit_path",     "")
    _set(s.output_dir,         "output_dir",         "")
    _set(s.output_name,        "output_name",        "krea2_lora")
    _set(s.save_precision,     "save_precision",     "bf16")
    # データセット
    _set(s.dataset_mode,                "dataset_mode",                DATASET_MODE_GUI)
    _set(s.dataset_config_path,         "dataset_config_path",         "")
    _set(s.general_resolution,          "general_resolution",          1024)
    _set(s.general_caption_extension,   "general_caption_extension",   ".txt")
    _set(s.general_enable_bucket,       "general_enable_bucket",       True)
    _set(s.general_bucket_no_upscale,   "general_bucket_no_upscale",   True)
    _set(s.batch_size,                  "batch_size",                  1)
    _set(s.num_workers,                 "num_workers",                 2)
    _set(s.persistent_workers,          "persistent_workers",          True)
    # エントリ復元
    entries_data = data.get("dataset_entries", [])
    if entries_data:
        from .state import DatasetEntryVars
        from .dataset_config import DatasetEntry
        s.dataset_entries.clear()
        for ed in entries_data:
            ev = DatasetEntryVars()
            ev.from_entry(DatasetEntry(
                image_dir         = ed.get("image_dir", ""),
                cache_directory   = ed.get("cache_directory", ""),
                num_repeats       = int(ed.get("num_repeats", 1)),
                caption_extension = ed.get("caption_extension", ".txt"),
                resolution        = int(ed.get("resolution", 1024)),
                enable_bucket     = bool(ed.get("enable_bucket", True)),
                bucket_no_upscale = bool(ed.get("bucket_no_upscale", True)),
                batch_size        = int(ed.get("batch_size", 0)),
            ))
            s.dataset_entries.append(ev)
    if not s.dataset_entries:
        from .state import DatasetEntryVars
        s.dataset_entries.append(DatasetEntryVars())
    _set(s.network_dim,        "network_dim",        32)
    _set(s.network_alpha,      "network_alpha",      32.0)
    _set(s.lora_target,        "lora_target",        "all")
    _set(s.network_args,       "network_args",       "")
    _set(s.network_weights,    "network_weights",    "")
    _set(s.layer_input_mode,   "layer_input_mode",   "text")
    if "layer_groups" in data and isinstance(data["layer_groups"], dict):
        for key, val in data["layer_groups"].items():
            if key in s.layer_groups and isinstance(val, bool):
                s.layer_groups[key].set(val)
    _set(s.learning_rate,      "learning_rate",      "1e-4")
    _set(s.lr_scheduler,       "lr_scheduler",       "cosine_with_restarts")
    _set(s.lr_warmup_steps,    "lr_warmup_steps",    0)
    _set(s.optimizer_type,     "optimizer_type",     "AdamW8bit")
    _set(s.optimizer_args,     "optimizer_args",     "")
    _set(s.max_train_epochs,   "max_train_epochs",   16)
    _set(s.save_every_n_epochs,"save_every_n_epochs",1)
    _set(s.seed,               "seed",               "42")
    _set(s.mixed_precision,    "mixed_precision",    "bf16")
    _set(s.grad_accum,         "grad_accum",         1)
    _set(s.max_grad_norm,      "max_grad_norm",      1.0)
    _set(s.gradient_checkpointing, "gradient_checkpointing", True)
    _set(s.timestep_sampling,  "timestep_sampling",  "shift")
    _set(s.discrete_flow_shift,"discrete_flow_shift",2.5)
    _set(s.weighting_scheme,   "weighting_scheme",   "none")
    _set(s.attn_mode,          "attn_mode",          "sdpa")
    _set(s.split_attn,         "split_attn",         False)
    _set(s.blocks_to_swap,     "blocks_to_swap",     0)
    _set(s.fp8_base,           "fp8_base",           False)
    _set(s.fp8_scaled,         "fp8_scaled",         False)
    _set(s.block_swap_h2d_only,"block_swap_h2d_only",False)
    _set(s.use_pinned_memory,  "use_pinned_memory",  False)
    _set(s.block_swap_ring_size,"block_swap_ring_size",2)
    _set(s.cpu_offload_checkpointing,"cpu_offload_checkpointing",False)
    _set(s.compile,            "compile",            False)
    _set(s.turbo_dit_cache,    "turbo_dit_cache",    False)
    _set(s.sample_enabled,     "sample_enabled",     False)
    _set(s.sample_every_n_epochs,"sample_every_n_epochs","1")
    _set(s.sample_prompt,      "sample_prompt",      "")
    _set(s.sample_negative_prompt,"sample_negative_prompt","")
    _set(s.sample_b_enabled,   "sample_b_enabled",   False)
    _set(s.sample_b_prompt,    "sample_b_prompt",    "")
    _set(s.sample_b_negative_prompt,"sample_b_negative_prompt","")
    _set(s.sample_width,       "sample_width",       1024)
    _set(s.sample_height,      "sample_height",      1024)
    _set(s.sample_steps,       "sample_steps",       28)
    _set(s.sample_guidance_scale,"sample_guidance_scale",5.5)
    # データセットエントリ変更を UI に通知（tab_dataset の _rebuild を呼ぶ）
    s.notify_dataset_reload()


def save_to_file(s: TrainState, preset_dir: Path, name: str) -> str | None:
    """プリセットを JSON ファイルに保存する。エラー時はメッセージを返す。"""
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    if not safe_name:
        return "プリセット名が空です。"
    preset_dir.mkdir(parents=True, exist_ok=True)
    dest = preset_dir / f"{safe_name}.json"
    try:
        dest.write_text(
            json.dumps(collect_preset(s), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        return f"保存失敗: {exc}"
    return None


def load_from_file(s: TrainState, path: Path) -> str | None:
    """JSON ファイルからプリセットを読み込む。エラー時はメッセージを返す。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"読み込み失敗: {exc}"
    apply_preset(s, data)
    return None


def list_presets(preset_dir: Path) -> list[str]:
    """preset_dir 内の JSON ファイル名（拡張子なし）を昇順で返す。"""
    if not preset_dir.exists():
        return []
    return sorted(p.stem for p in preset_dir.glob("*.json"))
