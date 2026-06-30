"""app/state.py — Krea2 LoRA 学習タブの状態オブジェクト。

全 tk.Variable をここで宣言・初期化し、タブビルダーや
コマンドビルダーは state を受け取るだけにする。
グラフデータは _graph_points に集約し、runner と monitor のキュー競合を回避する。
"""
from __future__ import annotations

import datetime
import queue
import subprocess
import threading
import tkinter as tk
from typing import Callable

from .config import AppPaths
from .dataset_config import DatasetEntry

# ── 選択肢定数 ──────────────────────────────────────────────────────────────
OPTIMIZERS: tuple[str, ...] = (
    "AdamW", "AdamW8bit", "Adafactor",
    "DAdaptAdam", "DAdaptAdaGrad", "DAdaptSGD",
    "Lion", "Prodigy", "CAME",
)
LR_SCHEDULERS: tuple[str, ...] = (
    "constant", "constant_with_warmup", "cosine",
    "cosine_with_restarts", "linear", "polynomial",
)
PRECISIONS: tuple[str, ...] = ("bf16", "fp16", "fp32")
TIMESTEP_SAMPLINGS: tuple[str, ...] = (
    "shift", "krea2_shift", "flux_shift", "sigmoid", "uniform",
)
WEIGHTING_SCHEMES: tuple[str, ...] = ("none", "sigma_sqrt", "cosmap")
ATTN_MODES: tuple[str, ...] = ("sdpa", "flash", "sageattn", "xformers")
LORA_TARGETS: tuple[str, ...] = ("all", "attention_only", "custom")

# カスタム選択時のサブモード
LAYER_INPUT_MODES: tuple[str, ...] = ("text", "group")

# 層グループ定義: (group_key, label, exclude_pattern)
# exclude_pattern は lora.py の re.fullmatch に渡す original_name パターン。
LAYER_GROUP_DEFS: tuple[tuple[str, str, str], ...] = (
    ("first",            "first（入力投影）",
     r"first"),
    ("blocks_0_9_attn",  "blocks.0〜9  Attention",
     r"blocks\.[0-9]\.attn\..*"),
    ("blocks_0_9_mlp",   "blocks.0〜9  MLP",
     r"blocks\.[0-9]\.mlp\..*"),
    ("blocks_10_19_attn","blocks.10〜19 Attention",
     r"blocks\.(1[0-9])\.attn\..*"),
    ("blocks_10_19_mlp", "blocks.10〜19 MLP",
     r"blocks\.(1[0-9])\.mlp\..*"),
    ("blocks_20_27_attn","blocks.20〜27 Attention",
     r"blocks\.(2[0-7])\.attn\..*"),
    ("blocks_20_27_mlp", "blocks.20〜27 MLP",
     r"blocks\.(2[0-7])\.mlp\..*"),
    ("txtfusion",        "txtfusion.*（テキスト融合）",
     r"txtfusion\..*"),
    ("tmlp_txtmlp",      "tmlp / txtmlp / tproj（時刻・テキスト投影）",
     r"(tmlp|txtmlp|tproj)\..*"),
    ("last_linear",      "last.linear（出力投影）",
     r"last\.linear"),
)


# データセット入力モード
DATASET_MODE_GUI  = "gui"    # GUI 入力 → toml 自動生成（デフォルト）
DATASET_MODE_TOML = "toml"   # 既存 toml を直接指定

SAMPLE_FIXED_SEED: int = 42


class DatasetEntryVars:
    """1 データセットエントリ分の tk.Variable 群。

    musubi-tuner スキーマに存在するキーのみ保持する。
    sd-scripts 固有の shuffle_caption / keep_tokens / flip_aug /
    min_bucket_reso / max_bucket_reso は存在しない。
    """

    def __init__(self) -> None:
        self.image_dir         = tk.StringVar()
        self.cache_directory   = tk.StringVar()
        self.num_repeats       = tk.IntVar(value=1)
        self.caption_extension = tk.StringVar(value=".txt")
        self.resolution        = tk.IntVar(value=1024)
        self.enable_bucket     = tk.BooleanVar(value=True)
        self.bucket_no_upscale = tk.BooleanVar(value=True)
        self.batch_size        = tk.IntVar(value=0)    # 0 = general に委ねる

    def to_entry(self) -> DatasetEntry:
        """tk.Variable の値を DatasetEntry に変換する。"""
        return DatasetEntry(
            image_dir         = self.image_dir.get(),
            cache_directory   = self.cache_directory.get(),
            num_repeats       = self.num_repeats.get(),
            caption_extension = self.caption_extension.get(),
            resolution        = self.resolution.get(),
            enable_bucket     = self.enable_bucket.get(),
            bucket_no_upscale = self.bucket_no_upscale.get(),
            batch_size        = self.batch_size.get(),
        )

    def from_entry(self, e: DatasetEntry) -> None:
        """DatasetEntry の値を tk.Variable に反映する。"""
        self.image_dir.set(e.image_dir)
        self.cache_directory.set(e.cache_directory)
        self.num_repeats.set(e.num_repeats)
        self.caption_extension.set(e.caption_extension)
        self.resolution.set(e.resolution)
        self.enable_bucket.set(e.enable_bucket)
        self.bucket_no_upscale.set(e.bucket_no_upscale)
        self.batch_size.set(e.batch_size)


class TrainState:
    """Krea2 LoRA 学習タブの全状態を保持する。"""

    def __init__(
        self,
        paths: AppPaths,
        log_fn: Callable[[str], None],
    ) -> None:
        self.paths  = paths
        self.log_fn = log_fn

        # ── プロセス管理 ──────────────────────────────────────
        self._proc: subprocess.Popen | None = None
        self._log_queue: queue.Queue[str]   = queue.Queue()
        self._log_widgets: list[tk.Text]    = []
        self._log_drain_started: bool       = False

        # ── グラフデータ（runner → monitor の単方向書き込み）──
        # 各タプルは (step, loss, lr)。loss は train avr_loss（musubi-tuner には
        # validation loss の出力がないため val loss は扱わない）。
        self._graph_points: list[tuple[int, float, float]] = []
        self._graph_lock: threading.Lock = threading.Lock()

        # grad_norm 用バッファ: (step, grad_norm)
        self._grad_norm_points: list[tuple[int, float]] = []
        self._grad_norm_lock: threading.Lock = threading.Lock()

        # epoch 進捗（最新値のみ保持）
        self._current_epoch: int = 0
        self._epoch_lock: threading.Lock = threading.Lock()

        # 学習開始時刻（ETA 計算用。launch_training() が設定する）
        self.training_start_time: datetime.datetime | None = None

        # ── モデルパス ────────────────────────────────────────
        self.dit_path           = tk.StringVar()
        self.vae_path           = tk.StringVar()
        self.text_encoder_path  = tk.StringVar()
        self.turbo_dit_path     = tk.StringVar()
        self.output_dir         = tk.StringVar(value=str(paths.output))
        self.output_name        = tk.StringVar(value="krea2_lora")
        self.save_precision     = tk.StringVar(value="bf16")

        # ── データセット（モード切替）────────────────────────
        self.dataset_mode     = tk.StringVar(value=DATASET_MODE_GUI)

        # [general] セクションの共通設定（musubi-tuner スキーマ）
        self.general_resolution        = tk.IntVar(value=1024)
        self.general_caption_extension = tk.StringVar(value=".txt")
        self.general_enable_bucket     = tk.BooleanVar(value=True)
        self.general_bucket_no_upscale = tk.BooleanVar(value=True)

        # GUI 入力モード: 複数エントリをリストで保持（初期1件）
        self.dataset_entries: list[DatasetEntryVars] = [DatasetEntryVars()]
        # 汎用バッチ/ワーカー設定（両モード共通）
        self.batch_size           = tk.IntVar(value=1)
        self.num_workers          = tk.IntVar(value=2)
        self.persistent_workers   = tk.BooleanVar(value=True)

        # TOML 直接指定モード
        self.dataset_config_path  = tk.StringVar()   # 既存 toml ファイルパス

        # ── ネットワーク ─────────────────────────────────────
        self.network_dim        = tk.IntVar(value=32)
        self.network_alpha      = tk.DoubleVar(value=32.0)
        self.lora_target        = tk.StringVar(value="all")
        self.network_args       = tk.StringVar(value="")
        self.network_weights    = tk.StringVar()

        # ── 層グループ選択（カスタム時サブモード）───────────
        # "text"  : network_args テキスト直接入力
        # "group" : 層グループ チェックボックス UI
        # tab_network.py が両モードを排他制御する（Entry の editable/readonly 切替）。
        self.layer_input_mode   = tk.StringVar(value="text")
        # 各グループのオン/オフ（True=学習対象に含める）
        self.layer_groups: dict[str, tk.BooleanVar] = {
            "first":             tk.BooleanVar(value=True),
            "blocks_0_9_attn":   tk.BooleanVar(value=True),
            "blocks_0_9_mlp":    tk.BooleanVar(value=True),
            "blocks_10_19_attn": tk.BooleanVar(value=True),
            "blocks_10_19_mlp":  tk.BooleanVar(value=True),
            "blocks_20_27_attn": tk.BooleanVar(value=True),
            "blocks_20_27_mlp":  tk.BooleanVar(value=True),
            "txtfusion":         tk.BooleanVar(value=True),
            "tmlp_txtmlp":       tk.BooleanVar(value=True),
            "last_linear":       tk.BooleanVar(value=True),
        }

        # ── 学習設定 ─────────────────────────────────────────
        self.learning_rate      = tk.StringVar(value="1e-4")
        self.lr_scheduler       = tk.StringVar(value="cosine_with_restarts")
        self.lr_warmup_steps    = tk.IntVar(value=0)
        self.optimizer_type     = tk.StringVar(value="AdamW8bit")
        self.optimizer_args     = tk.StringVar(value="")
        self.max_train_epochs   = tk.IntVar(value=16)
        self.save_every_n_epochs = tk.IntVar(value=1)
        self.seed               = tk.StringVar(value="42")
        self.mixed_precision    = tk.StringVar(value="bf16")
        self.grad_accum         = tk.IntVar(value=1)
        self.max_grad_norm      = tk.DoubleVar(value=1.0)
        self.gradient_checkpointing = tk.BooleanVar(value=True)

        # ── Krea2 詳細 ────────────────────────────────────────
        self.timestep_sampling   = tk.StringVar(value="shift")
        self.discrete_flow_shift = tk.DoubleVar(value=2.5)
        self.weighting_scheme    = tk.StringVar(value="none")

        # ── Attention ─────────────────────────────────────────
        self.attn_mode  = tk.StringVar(value="sdpa")
        self.split_attn = tk.BooleanVar(value=False)

        # ── メモリ最適化 ──────────────────────────────────────
        self.blocks_to_swap            = tk.IntVar(value=0)
        self.fp8_base                  = tk.BooleanVar(value=False)
        self.fp8_scaled                = tk.BooleanVar(value=False)
        self.block_swap_h2d_only       = tk.BooleanVar(value=False)
        self.use_pinned_memory         = tk.BooleanVar(value=False)
        self.block_swap_ring_size      = tk.IntVar(value=2)
        self.cpu_offload_checkpointing = tk.BooleanVar(value=False)
        self.compile                   = tk.BooleanVar(value=False)

        # ── Turbo ─────────────────────────────────────────────
        self.turbo_dit_cache = tk.BooleanVar(value=False)

        # ── サンプル生成 ──────────────────────────────────────
        self.sample_enabled           = tk.BooleanVar(value=False)
        self.sample_every_n_epochs    = tk.StringVar(value="1")
        self.sample_prompt            = tk.StringVar(value="")
        self.sample_negative_prompt   = tk.StringVar(value="")
        self.sample_b_enabled         = tk.BooleanVar(value=False)
        self.sample_b_prompt          = tk.StringVar(value="")
        self.sample_b_negative_prompt = tk.StringVar(value="")
        self.sample_width             = tk.IntVar(value=1024)
        self.sample_height            = tk.IntVar(value=1024)
        self.sample_steps             = tk.IntVar(value=28)
        self.sample_guidance_scale    = tk.DoubleVar(value=5.5)

        # ── ステータス ────────────────────────────────────────
        self.status_var = tk.StringVar(value="待機中")

    # ── コールバック ──────────────────────────────────────
    # プリセット読み込み後にデータセットタブのカード UI を再描画するために
    # tab_dataset.py が _rebuild 関数を登録する。
    # presest_manager.apply_preset → s.notify_dataset_reload() → _rebuild() の順で呼ばれる。
    def register_dataset_reload_callback(self, fn) -> None:
        """データセット UI 再描画コールバックを登録する（tab_dataset が呼ぶ）。"""
        if not hasattr(self, "_dataset_reload_callbacks"):
            self._dataset_reload_callbacks = []
        self._dataset_reload_callbacks.append(fn)

    def notify_dataset_reload(self) -> None:
        """データセットエントリが変化したことを UI に通知する。"""
        for fn in getattr(self, "_dataset_reload_callbacks", []):
            try:
                fn()
            except Exception:
                pass

    def is_running(self) -> bool:
        """学習プロセスが実行中かどうかを返す。"""
        return self._proc is not None and self._proc.poll() is None

    def push_graph_point(self, step: int, loss: float, lr: float) -> None:
        """グラフデータポイント（train loss / lr）をスレッドセーフに追加する。"""
        with self._graph_lock:
            self._graph_points.append((step, loss, float(lr)))

    def push_grad_norm_point(self, step: int, grad_norm: float) -> None:
        """grad_norm データポイントをスレッドセーフに追加する。"""
        with self._grad_norm_lock:
            self._grad_norm_points.append((step, float(grad_norm)))

    def push_epoch_point(self, epoch: int) -> None:
        """現在の epoch 番号をスレッドセーフに更新する。"""
        with self._epoch_lock:
            self._current_epoch = epoch

    def get_current_epoch(self) -> int:
        """現在の epoch 番号を返す。"""
        with self._epoch_lock:
            return self._current_epoch

    def reset_monitor_buffers(self) -> None:
        """モニターグラフ用バッファを全てクリアする（グラフリセット用）。"""
        with self._graph_lock:
            self._graph_points.clear()
        with self._grad_norm_lock:
            self._grad_norm_points.clear()
        with self._epoch_lock:
            self._current_epoch = 0
