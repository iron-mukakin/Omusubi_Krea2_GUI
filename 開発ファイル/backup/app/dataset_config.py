"""app/dataset_config.py — musubi-tuner 用 dataset_config.toml 生成。

musubi-tuner のスキーマ（config_utils.py ConfigSanitizer）に準拠する。

■ 正しい構造（musubi-tuner 形式）
  [general]
  resolution = 1024
  caption_extension = ".txt"
  batch_size = 1
  enable_bucket = true
  bucket_no_upscale = true

  [[datasets]]
  image_directory = "/path/to/images"
  cache_directory = "/path/to/cache"   # オプション
  num_repeats = 5                       # オプション

■ 誤った構造（sd-scripts 形式 → musubi-tuner では invalid）
  [[datasets]]
    [[datasets.subsets]]   ← musubi-tuner スキーマに存在しない

スキーマの詳細:
  DATASET_ASCENDABLE_SCHEMA（general / datasets 両方に置ける）:
    caption_extension, batch_size, num_repeats,
    resolution, enable_bucket, bucket_no_upscale

  IMAGE_DATASET_DISTINCT_SCHEMA（datasets 直下のみ）:
    image_directory, image_jsonl_file, cache_directory,
    control_directory, multiple_target, ...

  禁止キー（存在しない）:
    shuffle_caption, keep_tokens, flip_aug,
    min_bucket_reso, max_bucket_reso
    → musubi-tuner は bucket 境界を resolution から自動算出する
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── データクラス ─────────────────────────────────────────────────────────────

@dataclass
class DatasetEntry:
    """1 つの [[datasets]] エントリ。

    musubi-tuner のスキーマに存在するキーのみ保持する。
    """
    image_dir:         str   = ""
    cache_directory:   str   = ""    # オプション（空 = 未設定）
    num_repeats:       int   = 1

    # DATASET_ASCENDABLE_SCHEMA（general と重複してもよいが datasets 側に置いた場合上書き）
    caption_extension: str   = ".txt"
    resolution:        int   = 1024
    enable_bucket:     bool  = True
    bucket_no_upscale: bool  = True
    batch_size:        int   = 0     # 0 = general に委ねる（datasets 側に出力しない）


@dataclass
class DatasetConfig:
    """dataset_config.toml 全体。

    共通設定は [general] に、エントリ固有設定は [[datasets]] 直下に出力する。
    musubi-tuner スキーマにない shuffle_caption 等は toml に含めない。
    """
    # [general] キー
    resolution:        int   = 1024
    caption_extension: str   = ".txt"
    batch_size:        int   = 1
    enable_bucket:     bool  = True
    bucket_no_upscale: bool  = True

    # [[datasets]] リスト
    entries: list[DatasetEntry] = field(default_factory=list)


# ── TOML 生成 ────────────────────────────────────────────────────────────────

def generate_toml(cfg: DatasetConfig) -> str:
    """DatasetConfig を musubi-tuner 準拠の dataset_config.toml 文字列に変換する。"""
    lines: list[str] = []

    # [general] セクション
    lines += [
        "[general]",
        f"resolution = {cfg.resolution}",
        f'caption_extension = "{cfg.caption_extension}"',
        f"batch_size = {cfg.batch_size}",
        f"enable_bucket = {_bool(cfg.enable_bucket)}",
        f"bucket_no_upscale = {_bool(cfg.bucket_no_upscale)}",
        "",
    ]

    # [[datasets]] セクション（subsets なし・フラット構造）
    for entry in cfg.entries:
        lines.append("[[datasets]]")
        lines.append(f'image_directory = "{_esc(entry.image_dir)}"')

        if entry.cache_directory.strip():
            lines.append(f'cache_directory = "{_esc(entry.cache_directory)}"')

        if entry.num_repeats != 1:
            lines.append(f"num_repeats = {entry.num_repeats}")

        # caption_extension が general と異なる場合のみ出力
        if entry.caption_extension != cfg.caption_extension:
            lines.append(f'caption_extension = "{entry.caption_extension}"')

        # resolution が general と異なる場合のみ出力
        if entry.resolution != cfg.resolution:
            lines.append(f"resolution = {entry.resolution}")

        # batch_size が 0（general 委任）以外かつ general と異なる場合のみ出力
        if entry.batch_size > 0 and entry.batch_size != cfg.batch_size:
            lines.append(f"batch_size = {entry.batch_size}")

        lines.append("")

    return "\n".join(lines)


def write_toml(cfg: DatasetConfig, configs_dir: Path, stem: str = "dataset") -> Path:
    """toml を configs_dir/stem.toml に書き出してパスを返す。"""
    configs_dir.mkdir(parents=True, exist_ok=True)
    dest = configs_dir / f"{stem}.toml"
    dest.write_text(generate_toml(cfg), encoding="utf-8", newline="\n")
    return dest


# ── 内部ヘルパー ─────────────────────────────────────────────────────────────

def _bool(v: bool) -> str:
    return "true" if v else "false"


def _esc(s: str) -> str:
    """TOML 文字列用にバックスラッシュをエスケープする。"""
    return s.replace("\\", "\\\\")
