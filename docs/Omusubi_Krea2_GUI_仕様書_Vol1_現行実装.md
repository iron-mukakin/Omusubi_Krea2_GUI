# Omusubi_Krea2_GUI — アプリケーション仕様書 Vol.1
**現行実装仕様書 / AIエージェント引き継ぎ用**
バージョン: 1.0.0（2026-06-28 確定）

---

## 1. プロジェクト概要

### 1.1 アプリケーション名・目的
- **名称**: Omusubi_Krea2_GUI
- **目的**: kohya-ss/musubi-tuner を使用した Krea2 LoRA 学習の Windows GUI フロントエンド
- **対象モデル**: Krea2（Single Stream MMDiT）
- **フレームワーク**: Python 3.10〜3.12 + tkinter（標準ライブラリのみ）
- **プラットフォーム**: Windows 専用（multiprocessing spawn 対応設計）

### 1.2 基本方針
- musubi-tuner はオリジナルを `git clone` して流用（アップデート対応）
- 仮想環境は `musubi-tuner/.venv/` に設置（musubi-tuner 単体でも動作）
- GUI コードは `app/` フォルダに格納、musubi-tuner とは疎結合
- `_krea2_launcher.py` を経由して `krea2_train_network.py` を起動

---

## 2. フォルダ・ファイル構成

```
<GUIルート>/                   ← Omusubi_Krea2_GUI/
├── setup.bat                  ← 初回セットアップ（clone + venv + pip）
├── start.bat                  ← GUI 起動（musubi-tuner/.venv を使用）
├── apply_fix_*.py             ← パッチスクリプト群（バグ修正用）
│
├── musubi-tuner/              ← git clone（オリジナル流用・手動編集禁止）
│   ├── .venv/                 ← 仮想環境（Python + PyTorch + musubi-tuner deps）
│   └── src/musubi_tuner/      ← 学習スクリプト群
│       ├── krea2_train_network.py
│       ├── krea2_cache_latents.py
│       └── krea2_cache_text_encoder_outputs.py
│
├── app/                       ← GUI アプリコード（20 ファイル, 3073 行）
│   ├── __init__.py
│   ├── main.py                ← エントリポイント
│   ├── config.py              ← AppPaths（パス管理）
│   ├── state.py               ← TrainState（全 tk.Variable）
│   ├── cmd_builder.py         ← CLI コマンド生成 + launcher 生成
│   ├── runner.py              ← subprocess 管理・ログ転送
│   ├── run_panel.py           ← 実行パネル UI
│   ├── widgets.py             ← 共通 Tk ウィジェットヘルパー
│   ├── dataset_config.py      ← dataset_config.toml 生成
│   ├── preset_manager.py      ← プリセット収集・適用・保存・読込
│   ├── tab_lora.py            ← LoRA 学習メインタブ（オーケストレーター）
│   ├── tab_model.py           ← [モデル] タブ
│   ├── tab_dataset.py         ← [データセット] タブ
│   ├── tab_cache.py           ← [キャッシュ生成] タブ
│   ├── tab_network.py         ← [ネットワーク] タブ
│   ├── tab_train.py           ← [学習設定] タブ
│   ├── tab_advanced.py        ← [詳細・最適化] タブ
│   ├── tab_sample.py          ← [サンプル生成] タブ
│   ├── tab_monitor.py         ← [モニター] タブ（loss/lr グラフ）
│   └── tab_preset.py          ← [プリセット] タブ
│
├── preset/
│   └── krea2/
│       ├── default_krea2.json
│       └── low_vram_blockswap.json
│
├── configs/                   ← GUI が自動生成する dataset_config.toml
├── output/                    ← 学習出力（LoRA .safetensors）
└── log/
    ├── krea2_train/           ← 学習ログ（YYYYMMDD_HHMMSS.txt）
    └── sample_gen/            ← サンプル生成画像
```

---

## 3. 起動フロー

```
start.bat
  └─ musubi-tuner/.venv/Scripts/python.exe -m app.main
       └─ main() [main.py]
            ├─ AppPaths.from_root()
            ├─ Tk root 生成・スタイル設定
            ├─ グローバルログ領域
            ├─ ttk.Notebook（グローバル）
            │   └─ [Krea2 LoRA 学習] タブ → build_lora_tab()
            └─ root.mainloop()
```

### 学習起動フロー

```
[▶ 実行] タブ → ▶ 学習開始 ボタン
  └─ run_panel._on_start()
       ├─ validate_inputs(s)          # バリデーション
       ├─ resolve_dataset_config(s)   # toml 生成 or 既存パス取得
       ├─ build_train_command(s, p)   # CLI コマンドリスト生成
       │   └─ write_wrapper_script()  # _krea2_launcher.py を動的生成
       └─ launch_training(s, cmd)     # バックグラウンドスレッドで起動
            └─ subprocess.Popen(
                 [python, -m, accelerate.commands.launch,
                  --mixed_precision, bf16,
                  _krea2_launcher.py,
                  --dit, ..., --vae, ..., ...]
               )
```

### _krea2_launcher.py（動的生成）

```python
# Windows spawn 対応: exec() 廃止、runpy.run_module() 採用
import sys, os, runpy
from multiprocessing import freeze_support

_SRC  = '<musubi-tuner/src の絶対パス>'
_ROOT = '<musubi-tuner/ の絶対パス>'

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_ROOT)

if __name__ == '__main__':           # spawn worker は通過しない
    freeze_support()
    runpy.run_module(
        'musubi_tuner.krea2_train_network',
        run_name='__main__',
        alter_sys=True,
    )
```

**spawn 安全性**: DataLoader worker が launcher を再 import しても
`__name__ != '__main__'` のため `run_module` は呼ばれない。

---

## 4. モジュール仕様

### 4.1 config.py — AppPaths

```python
@dataclass(frozen=True)
class AppPaths:
    root:             Path   # GUIルート
    app:              Path   # app/
    musubi:           Path   # musubi-tuner/
    musubi_src:       Path   # musubi-tuner/src/
    musubi_venv_py:   Path   # musubi-tuner/.venv/Scripts/python.exe
    preset:           Path   # preset/
    log:              Path   # log/
    output:           Path   # output/

    @classmethod
    def from_root(cls, root=None) -> AppPaths  # __file__ から自動解決
    def ensure_dirs(self) -> None               # 必要ディレクトリを作成
    def validate_musubi(self) -> str | None     # musubi-tuner 存在確認

    @property
    def krea2_train_script(self) -> Path        # krea2_train_network.py
    def krea2_cache_latents_script(self) -> Path
    def krea2_cache_te_script(self) -> Path
```

### 4.2 state.py — TrainState / DatasetEntryVars

#### DatasetEntryVars（1 [[datasets]] エントリ）

| 変数名 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `image_dir` | StringVar | "" | image_directory |
| `cache_directory` | StringVar | "" | キャッシュ保存先 |
| `num_repeats` | IntVar | 1 | 繰り返し回数 |
| `caption_extension` | StringVar | ".txt" | キャプション拡張子 |
| `resolution` | IntVar | 1024 | general 上書き用 |
| `enable_bucket` | BooleanVar | True | general 上書き用 |
| `bucket_no_upscale` | BooleanVar | True | general 上書き用 |
| `batch_size` | IntVar | 0 | 0=general 委任 |

メソッド: `to_entry() -> DatasetEntry`, `from_entry(e: DatasetEntry)`

#### TrainState の主要変数

| グループ | 変数名 | 型 | デフォルト |
|---|---|---|---|
| **モデル** | `dit_path` | StringVar | "" |
| | `vae_path` | StringVar | "" |
| | `text_encoder_path` | StringVar | "" |
| | `turbo_dit_path` | StringVar | "" |
| | `output_dir` | StringVar | paths.output |
| | `output_name` | StringVar | "krea2_lora" |
| | `save_precision` | StringVar | "bf16" |
| **データセット** | `dataset_mode` | StringVar | "gui" |
| | `general_resolution` | IntVar | 1024 |
| | `general_caption_extension` | StringVar | ".txt" |
| | `general_enable_bucket` | BooleanVar | True |
| | `general_bucket_no_upscale` | BooleanVar | True |
| | `batch_size` | IntVar | 1 |
| | `num_workers` | IntVar | 2 |
| | `persistent_workers` | BooleanVar | True |
| | `dataset_config_path` | StringVar | "" |
| | `dataset_entries` | list[DatasetEntryVars] | [1件] |
| **ネットワーク** | `network_dim` | IntVar | 32 |
| | `network_alpha` | DoubleVar | 32.0 |
| | `lora_target` | StringVar | "all" |
| | `network_args` | StringVar | "" |
| | `network_weights` | StringVar | "" |
| **学習設定** | `learning_rate` | StringVar | "1e-4" |
| | `lr_scheduler` | StringVar | "cosine_with_restarts" |
| | `optimizer_type` | StringVar | "AdamW8bit" |
| | `max_train_epochs` | IntVar | 16 |
| | `save_every_n_epochs` | IntVar | 1 |
| | `mixed_precision` | StringVar | "bf16" |
| | `gradient_checkpointing` | BooleanVar | True |
| **Krea2 詳細** | `timestep_sampling` | StringVar | "shift" |
| | `discrete_flow_shift` | DoubleVar | 2.5 |
| | `weighting_scheme` | StringVar | "none" |
| **Attention** | `attn_mode` | StringVar | "sdpa" |
| | `split_attn` | BooleanVar | False |
| **メモリ最適化** | `blocks_to_swap` | IntVar | 0 |
| | `fp8_base` | BooleanVar | False |
| | `fp8_scaled` | BooleanVar | False |
| | `block_swap_h2d_only` | BooleanVar | False |
| | `use_pinned_memory` | BooleanVar | False |
| | `block_swap_ring_size` | IntVar | 2 |
| | `turbo_dit_cache` | BooleanVar | False |
| **サンプル** | `sample_enabled` | BooleanVar | False |
| | `sample_prompt` | StringVar | "" |
| | `sample_width` | IntVar | 1024 |
| | `sample_steps` | IntVar | 28 |
| | `sample_guidance_scale` | DoubleVar | 5.5 |

#### TrainState のメソッド

```python
is_running() -> bool
push_graph_point(step, loss, lr)          # モニターグラフ用
register_dataset_reload_callback(fn)      # プリセット読込後の UI 再描画
notify_dataset_reload()                   # コールバックを全呼び出し
```

#### TrainState の内部属性（プライベート）

```python
_proc: subprocess.Popen | None           # 学習プロセス
_log_queue: queue.Queue[str]             # ログキュー（runner → ウィジェット）
_log_widgets: list[tk.Text]             # ログ出力先ウィジェット
_graph_points: list[tuple[int,float,float]]  # (step, loss, lr)
_graph_lock: threading.Lock
_log_drain_started: bool
_dataset_reload_callbacks: list[Callable]
```

### 4.3 dataset_config.py — DatasetConfig / DatasetEntry

musubi-tuner 準拠の `dataset_config.toml` を生成する。
**`[[datasets.subsets]]` は存在しない（sd-scripts との差異）。**

```python
@dataclass
class DatasetEntry:
    image_dir, cache_directory, num_repeats
    caption_extension, resolution, enable_bucket, bucket_no_upscale, batch_size

@dataclass
class DatasetConfig:
    resolution, caption_extension, batch_size        # → [general]
    enable_bucket, bucket_no_upscale                 # → [general]
    entries: list[DatasetEntry]                      # → [[datasets]]
```

**生成される toml の構造:**
```toml
[general]
resolution = 1024
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "E:/path/to/images"
cache_directory = "E:/path/to/cache"
num_repeats = 5
```

### 4.4 cmd_builder.py — コマンド生成

```python
write_wrapper_script(paths) -> Path    # _krea2_launcher.py を生成
validate_inputs(s) -> str | None       # 必須入力バリデーション
build_train_command(s, paths) -> list[str]  # accelerate launch コマンド生成
```

**バリデーション項目:**
- `dit_path`, `vae_path` の存在確認
- dataset: GUI モード→image_dir 存在確認 / TOML モード→ファイル存在確認
- `fp8_base=True` かつ `fp8_scaled=False` → エラー（Krea2 制約）
- `turbo_dit` と `blocks_to_swap > 0` の併用 → エラー
- `turbo_dit_cache` かつ `turbo_dit` 未設定 → エラー

**Krea2 固有の fp8 安全ガード:**
```python
# Bool フラグ展開後に追加で保証
if s.fp8_base.get() and "--fp8_scaled" not in cmd:
    cmd.append("--fp8_scaled")
```

**krea2_train_network.py に存在しない引数（送信禁止）:**
- `--train_batch_size` → toml の `batch_size` で設定
- `--max_data_loader_n_workers` → toml で設定

### 4.5 runner.py — プロセス管理

```python
launch_training(s, cmd, on_finish=None)  # バックグラウンドスレッドで起動
stop_training(s)                          # CTRL_BREAK_EVENT 送信
start_log_drain(widget_getter, s, root)  # after() ポーリングでログ転送
```

**グラフデータ分離**: ログキュー（`_log_queue`）は UI ウィジェット用、
グラフデータ（`_graph_points`）は `push_graph_point()` 経由で書き込み。
`tab_monitor.py` は `_graph_points` をポーリングして描画する（キュー競合なし）。

### 4.6 tab_dataset.py — データセットタブ

**モード切替:**
- `DATASET_MODE_GUI` (`"gui"`): GUI 入力 → `configs/<name>.toml` 自動生成
- `DATASET_MODE_TOML` (`"toml"`): 既存 toml ファイルを直接指定

**プリセット連携（オブザーバーパターン）:**
```
apply_preset() → notify_dataset_reload()
  → _on_preset_reload()
       ├─ _rebuild()       # エントリカード再描画
       └─ _switch_mode()   # モード切替 UI 同期
```

**公開関数:**
```python
resolve_dataset_config(s) -> str  # cmd_builder から呼ばれる。パス解決
```

### 4.7 preset_manager.py — プリセット管理

```python
collect_preset(s) -> dict          # TrainState → JSON dict
apply_preset(s, data)              # JSON dict → TrainState（+ notify）
save_to_file(s, preset_dir, name) -> str | None
load_from_file(s, path) -> str | None
list_presets(preset_dir) -> list[str]
```

プリセット保存先: `preset/krea2/<name>.json`

### 4.8 tab_advanced.py — 排他制御

**UI レベルの排他制御:**

| 制約 | 実装 |
|---|---|
| `turbo_dit` パス入力 → `blocks_to_swap` を 0 にリセット + disabled | `s.turbo_dit_path.trace_add("write", _on_turbo_path_change)` |
| `fp8_base` ON/OFF → `fp8_scaled` を同期 | `_on_fp8_toggle(s)` |
| `turbo_dit_cache` ON かつ `turbo_dit` 未設定 → 自動解除 | `_on_turbo_cache(s)` |

### 4.9 tab_cache.py — キャッシュ生成

**Step 1 — Latent キャッシュ:**
```
krea2_cache_latents.py --dataset_config <toml> --vae <vae> [--skip_existing]
```

**Step 2 — Text Encoder キャッシュ:**
```
krea2_cache_text_encoder_outputs.py
  --dataset_config <toml>
  --text_encoder <qwen3vl_bf16.safetensors>   ← fp8量子化版(comfy_quant)は不可
  --batch_size 1
  --text_encoder_dtype bfloat16
  [--skip_existing]
```

**注意**: Text Encoder は ComfyUI の fp8 量子化版（`comfy_quant` キー含む）では
`RuntimeError: unexpected keys` が発生する。bf16 非量子化版が必要。

### 4.10 widgets.py — 共通ウィジェット

```python
label_entry_row(parent, row, label, var, ...)     → ttk.Entry
label_spinbox_row(parent, row, label, var, ...)   → ttk.Spinbox
label_combobox_row(parent, row, label, var, ...)  → ttk.Combobox
browse_file_row(parent, row, label, var, ...)     # Entry + Browse ボタン
browse_dir_row(parent, row, label, var, ...)      # Entry + Browse ボタン
labeled_frame(parent, text, **pack_kw)            → ttk.LabelFrame（pack 済み）
attach_log_widget(parent, log_widgets, height)    → tk.Text
```

**レイアウト規則（pack/grid 混在禁止）:**
- `labeled_frame()` が返す `LabelFrame` は `parent` に `pack` 済み
- `LabelFrame` の子ウィジェットは `grid` で配置
- 同一 `parent` に `pack` と `grid` を混在させない

---

## 5. LoRA タブ構成

```
ttk.Notebook（グローバル）
└─ [Krea2 LoRA 学習] → tab_lora.py: build_lora_tab()
     └─ ttk.Notebook（LoRA 内）
          ├─ [モデル]       tab_model.py
          ├─ [データセット]  tab_dataset.py
          ├─ [キャッシュ生成] tab_cache.py
          ├─ [ネットワーク]  tab_network.py
          ├─ [学習設定]     tab_train.py
          ├─ [詳細・最適化]  tab_advanced.py
          ├─ [サンプル生成]  tab_sample.py
          ├─ [モニター]     tab_monitor.py
          ├─ [プリセット]   tab_preset.py
          └─ [▶ 実行]      run_panel.py
```

---

## 6. プリセット JSON スキーマ

```json
{
  "dit_path": "",
  "vae_path": "",
  "text_encoder_path": "",
  "turbo_dit_path": "",
  "output_dir": "",
  "output_name": "krea2_lora",
  "save_precision": "bf16",
  "dataset_mode": "gui",
  "dataset_config_path": "",
  "general_resolution": 1024,
  "general_caption_extension": ".txt",
  "general_enable_bucket": true,
  "general_bucket_no_upscale": true,
  "batch_size": 1,
  "num_workers": 2,
  "persistent_workers": true,
  "dataset_entries": [
    {
      "image_dir": "",
      "cache_directory": "",
      "num_repeats": 1,
      "caption_extension": ".txt",
      "resolution": 1024,
      "enable_bucket": true,
      "bucket_no_upscale": true,
      "batch_size": 0
    }
  ],
  "network_dim": 32,
  "network_alpha": 32.0,
  "lora_target": "all",
  "network_args": "",
  "network_weights": "",
  "learning_rate": "1e-4",
  "lr_scheduler": "cosine_with_restarts",
  "lr_warmup_steps": 0,
  "optimizer_type": "AdamW8bit",
  "optimizer_args": "",
  "max_train_epochs": 16,
  "save_every_n_epochs": 1,
  "seed": "42",
  "mixed_precision": "bf16",
  "grad_accum": 1,
  "max_grad_norm": 1.0,
  "gradient_checkpointing": true,
  "timestep_sampling": "shift",
  "discrete_flow_shift": 2.5,
  "weighting_scheme": "none",
  "attn_mode": "sdpa",
  "split_attn": false,
  "blocks_to_swap": 0,
  "fp8_base": false,
  "fp8_scaled": false,
  "block_swap_h2d_only": false,
  "use_pinned_memory": false,
  "block_swap_ring_size": 2,
  "cpu_offload_checkpointing": false,
  "compile": false,
  "turbo_dit_cache": false,
  "sample_enabled": false,
  "sample_every_n_epochs": "1",
  "sample_prompt": "",
  "sample_negative_prompt": "",
  "sample_b_enabled": false,
  "sample_b_prompt": "",
  "sample_b_negative_prompt": "",
  "sample_width": 1024,
  "sample_height": 1024,
  "sample_steps": 28,
  "sample_guidance_scale": 5.5
}
```

---

## 7. 生成される CLI コマンド例

```bash
python -m accelerate.commands.launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  app/_krea2_launcher.py \
  --dit E:/path/raw.safetensors \
  --vae E:/path/vae.safetensors \
  --dataset_config E:/Omusubi_Krea2_GUI/configs/krea2_lora.toml \
  --network_module networks.lora_krea2 \
  --network_dim 32 \
  --network_alpha 32.0 \
  --output_dir E:/Omusubi_Krea2_GUI/output \
  --output_name krea2_lora \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 0 \
  --optimizer_type AdamW8bit \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --gradient_accumulation_steps 1 \
  --max_grad_norm 1.0 \
  --timestep_sampling shift \
  --discrete_flow_shift 2.5 \
  --weighting_scheme none \
  --seed 42 \
  --sdpa \
  --gradient_checkpointing \
  --persistent_data_loader_workers \
  --fp8_base --fp8_scaled \
  --blocks_to_swap 24 \
  --block_swap_ring_size 2 \
  --block_swap_h2d_only
```

---

## 8. 既知の制約・注意事項

| 項目 | 内容 |
|---|---|
| fp8 Text Encoder | ComfyUI の fp8 量子化版（comfy_quant キー）は TE キャッシュ生成に使用不可。bf16 版が必要 |
| dataset_config.toml | sd-scripts 形式（`[[datasets.subsets]]`）は不可。musubi-tuner フラット形式のみ |
| turbo_dit × blocks_to_swap | 同時使用不可（UI レベルでブロック済み） |
| fp8_base 単体 | Krea2 は fp8_scaled とのセット必須（UI 連動 + 安全ガード実装済み） |
| train_batch_size | CLI 引数として渡さない（toml に記述済み） |
| Windows spawn | DataLoader `num_workers > 0` 対応済み（launcher に freeze_support() + if __name__ ガード）|

---

## 9. 依存ライブラリ

### musubi-tuner/.venv（学習環境）
- Python 3.10〜3.12
- torch >= 2.7 (cu128 推奨) / cu124 フォールバック
- accelerate 1.6.0
- transformers 4.57.6
- safetensors 0.4.5
- einops 0.7.0
- bitsandbytes（AdamW8bit 用）

### GUI 追加
- matplotlib（モニターグラフ用・オプション）
- Pillow（サンプル画像ギャラリー用・オプション）
- tkinter（Python 標準）
