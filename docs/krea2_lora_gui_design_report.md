# Krea2 LoRA GUI 設計レポート

## 1. ファイル調査結果

### 1.1 参照元: Anima GUIの構成

| ファイル | 役割 |
|---|---|
| `lora_train.py` | LoRA学習タブ本体（2060行）。`_TrainState`で変数管理、各タブビルダー関数、`_build_command()`でCLI生成 |
| `config.py` | `AppPaths`（相対パス管理）、`MergeOptions` dataclass |
| `setup_start.bat` | venv作成、pip install、アプリ起動の一連のバッチ |

**lora_train.py の設計パターン（移植対象）:**
- `_TrainState` クラスに全 `tk.Variable` を集約
- タブ毎のビルダー関数 `_build_XXX_tab()` に分離
- `_build_command()` で `accelerate launch` コマンドリスト生成
- `subprocess.Popen` + `threading` でバックグラウンド実行
- `queue.Queue` + `after()` ポーリングでログをGUIへ反映
- プリセットは JSON ファイル保存（`preset/` フォルダ）

---

### 1.2 Krea2 固有の CLI インターフェース

#### 必須引数（`krea2_train_network.py`）

```
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16
  src/musubi_tuner/krea2_train_network.py
  --dit           <RAW .safetensors>        # 必須
  --vae           <qwen_image_vae.safetensors>  # 必須
  --dataset_config <toml>                  # 必須
  --network_module networks.lora_krea2     # 固定値（Krea2専用）
  --network_dim    32
  --network_alpha  32
  --output_dir    <dir>
  --output_name   <name>
```

#### テキストエンコーダー（サンプル生成時のみ必要）
```
  --text_encoder  <qwen3vl_4b_bf16.safetensors>
```

#### Krea2固有の追加オプション（`krea2_setup_parser()`）

| 引数 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `--fp8_scaled` | flag | off | dynamic scaled fp8（`--fp8_base` も必須） |
| `--text_encoder` | str | None | Qwen3-VL-4B パス（サンプル生成時のみ） |
| `--turbo_dit` | str | None | TurboのDiTパス（サンプル生成でRAW→Turbo） |
| `--turbo_dit_cache` | flag | off | Turboを起動時にCPU常駐（M1モード） |

#### 制約（排他・依存関係）
- `--fp8_base` 単体は不可 → `--fp8_scaled` と必ずセット
- `--turbo_dit_cache` は `--turbo_dit` が必要
- `--turbo_dit` と `--blocks_to_swap` は併用不可

#### 共通オプション（`setup_parser_common()` / HunyuanVideo共通）

| カテゴリ | 主要引数 |
|---|---|
| **タイムステップ** | `--timestep_sampling` {shift, krea2_shift, flux_shift, sigmoid, uniform}, `--discrete_flow_shift` (推奨2.5 @1024px) |
| **重み付け** | `--weighting_scheme` {none, sigma_sqrt, cosmap} |
| **Attention** | `--sdpa` / `--flash_attn` / `--sage_attn` / `--xformers`, `--split_attn` |
| **Block Swap** | `--blocks_to_swap` 0-26, `--use_pinned_memory_for_block_swap`, `--block_swap_h2d_only`, `--block_swap_ring_size` |
| **コンパイル** | `--compile` |
| **メモリ** | `--gradient_checkpointing`, `--fp8_base`, `--cpu_offload_checkpointing` |
| **データセット** | `--dataset_config` (toml), `--max_data_loader_n_workers`, `--persistent_data_loader_workers` |
| **ネットワーク** | `--network_dim`, `--network_alpha`, `--network_args` (exclude/include_patterns) |
| **学習率** | `--learning_rate`, `--lr_scheduler`, `--lr_warmup_steps`, `--optimizer_type` |
| **エポック** | `--max_train_epochs`, `--save_every_n_epochs`, `--seed` |
| **サンプル** | `--sample_prompts`, `--sample_every_n_epochs` |

#### LoRAターゲット選択（`--network_args`）
```bash
# デフォルト: 全264 Linear層（attention + MLP + txtfusion + projection）
# Attentionのみ:
--network_args "exclude_patterns=['.*\.mlp\..*','first','last\.linear','tmlp\..*','txtmlp\..*','tproj\.1','txtfusion\..*']"
# 任意サブセット:
--network_args "exclude_patterns=['.*']" "include_patterns=[...]"
```

#### 事前キャッシングスクリプト
```bash
# Latent
python src/musubi_tuner/krea2_cache_latents.py --dataset_config <toml> --vae <vae>

# Text Encoder
python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \
    --dataset_config <toml> --text_encoder <qwen3vl> --batch_size 1
```

---

## 2. フォルダ構成

```
<GUIアプリルート>/              ← バッチ・pyproject.toml
├── setup.bat                   ← venv作成 + pip install
├── run.bat                     ← アプリ起動
├── pyproject.toml / requirements.txt
├── .venv/                      ← 仮想環境（GUI root直下）
├── musubi-tuner/               ← git clone（オリジナル流用）
│   └── src/musubi_tuner/
│       ├── krea2_train_network.py
│       ├── krea2_cache_latents.py
│       ├── krea2_cache_text_encoder_outputs.py
│       └── ...
├── app/                        ← GUIアプリコード
│   ├── __init__.py
│   ├── main.py                 ← エントリポイント（Tk root 生成）
│   ├── config.py               ← AppPaths（animaのconfig.py移植）
│   ├── krea2_lora_tab.py       ← Krea2 LoRA学習タブ本体
│   └── i18n.py                 ← 国際化（必要なら）
├── preset/                     ← プリセット JSON
│   └── default_krea2.json
└── log/                        ← 学習ログ（自動生成）
    └── krea2_train/
```

---

## 3. GUIタブ構成（Phase 1 最小実装）

```
Notebook
├── [1] モデル
│     DiT (RAW), VAE, Text Encoder(opt), Turbo DiT(opt)
│     出力ディレクトリ, 出力名, 保存精度
├── [2] データセット
│     dataset_config (toml), batch_size, num_workers,
│     persistent_data_loader_workers
├── [3] ネットワーク
│     network_dim, network_alpha
│     LoRAターゲット: All / Attention-Only / Custom(network_args)
│     network_weights (継続学習)
├── [4] 学習設定
│     learning_rate, lr_scheduler, lr_warmup_steps
│     optimizer_type, optimizer_args
│     max_train_epochs, save_every_n_epochs, seed
│     gradient_accumulation_steps, max_grad_norm, mixed_precision
│     gradient_checkpointing
├── [5] 詳細・メモリ最適化
│     timestep_sampling, discrete_flow_shift, weighting_scheme
│     attn_mode (sdpa/flash/sage/xformers), split_attn
│     blocks_to_swap (0-26), fp8_base+fp8_scaled (連動)
│     block_swap_h2d_only, use_pinned_memory_for_block_swap, block_swap_ring_size
│     cpu_offload_checkpointing, compile
├── [6] サンプル生成
│     sample_every_n_epochs, sample_prompts(file or text)
│     turbo_dit, turbo_dit_cache (連動ロック: blocks_to_swapと排他)
├── [7] プリセット           ← animaのプリセットタブを移植
│     リスト / 保存 / 読込 / 削除 / Export / Import
└── [実行パネル] (各主要タブ下部)
      コマンドプレビュー, 開始, 停止, ステータス, ログ
```

---

## 4. コマンド生成の差分（anima → Krea2）

| 項目 | Anima | Krea2 |
|---|---|---|
| 学習スクリプト | `anima_train_network.py` | `krea2_train_network.py` |
| ラップ方法 | `_gui_train_wrapper.py` 生成 | 同方式（パス変更のみ） |
| `--dit` | `--pretrained_model_name_or_path` | `--dit` |
| テキストエンコーダー | `--qwen3` | `--text_encoder`（サンプル時のみ） |
| `--network_module` | `networks.lora` | `networks.lora_krea2`（固定） |
| fp8 | 単独 `--fp8_base` | `--fp8_base --fp8_scaled` 必ずペア |
| turbo | なし | `--turbo_dit`, `--turbo_dit_cache` |
| timestep_sampling | sigmoid/sigma/uniform/shift | + `krea2_shift` 追加 |
| `--dataset_config` | なし（`--train_data_dir`） | **必須**（tomlファイル） |

---

## 5. 実装上の注意事項

### 5.1 dataset_config (toml) の扱い
Krea2はAnimaと異なり`--dataset_config`が必須（tomlファイル）。  
`--train_data_dir`は使用しない。GUIではtomlファイルをブラウズして指定。

### 5.2 fp8の連動バリデーション
`fp8_scaled`ON → `fp8_base`も強制ON（またはUIで連動チェック）。  
`fp8_base`単体は`krea2_train_network.py`がValueErrorで弾く。

### 5.3 turbo_ditとblocks_to_swapの排他
UI側で`turbo_dit`パスが入力されている場合、`blocks_to_swap > 0`にしようとしたら警告またはロック。

### 5.4 wrapperスクリプトのパス
```python
musubi_root = app_paths.root / "musubi-tuner"
train_script = musubi_root / "src" / "musubi_tuner" / "krea2_train_network.py"
src_path = musubi_root / "src"  # PYTHONPATH に追加
```

### 5.5 accelerate launchの呼び出し
```python
cmd = [
    sys.executable, "-m", "accelerate.commands.launch",
    "--num_cpu_threads_per_process", "1",
    "--mixed_precision", mixed_precision,
    str(wrapper_script),
    "--dit", dit_path,
    "--vae", vae_path,
    "--dataset_config", dataset_config_path,
    "--network_module", "networks.lora_krea2",
    ...
]
```

### 5.6 仮想環境の依存関係
```
# musubi-tuner の依存 + GUI用
musubi-tunerのrequirements (accelerate, torch, transformers, safetensors, einops, ...)
+ tkinter (Python標準)
# Windowsではtk/tcl同梱のため追加不要
```

---

## 6. Phase 1 実装ファイルリスト

| ファイル | 内容 |
|---|---|
| `setup.bat` | venv作成、musubi-tunerのgit clone、pip install |
| `run.bat` | `.venv\Scripts\python.exe app\main.py` |
| `app/config.py` | `AppPaths.from_root()` (musubi-tunerパス含む) |
| `app/main.py` | Tk root、ウィンドウタイトル、ノートブック |
| `app/krea2_lora_tab.py` | `build_krea2_lora_tab()` 本体 |
| `preset/default_krea2.json` | デフォルトプリセット |

---

## 7. プリセットのキー設計（JSON）

animaのプリセット方式をそのまま踏襲。`_TrainState`の変数名をキーに使用。

```json
{
  "preset_name": "default_krea2_1024",
  "dit_path": "",
  "vae_path": "",
  "text_encoder_path": "",
  "turbo_dit_path": "",
  "output_dir": "",
  "output_name": "krea2_lora",
  "precision": "bf16",
  "mixed_precision": "bf16",
  "network_dim": 32,
  "network_alpha": 32,
  "network_target": "all",
  "learning_rate": "1e-4",
  "lr_scheduler": "cosine_with_restarts",
  "lr_warmup_steps": 0,
  "optimizer_type": "AdamW8bit",
  "max_train_epochs": 16,
  "save_every_n_epochs": 1,
  "seed": "42",
  "batch_size": 1,
  "num_workers": 2,
  "persistent_workers": true,
  "gradient_checkpointing": true,
  "grad_accum": 1,
  "max_grad_norm": 1.0,
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
  "sample_every_n_epochs": "1",
  "sample_prompts_file": ""
}
```

---

## 8. 未実装（Phase 2以降）

以下はanima由来の機能でKrea2側に相当するものがなく、GUIアプリ独自実装が必要:

- **階層学習（Layer LR）**: Krea2は`blocks.0`〜`blocks.27`の28ブロック + txtfusion。`exclude_patterns` / `include_patterns` で代替可能だが、スライダーUIはPhase 2
- **学習モニター**: loss/lr グラフ描画タブ
- **Entry Stopping / Validation Split**: GUIアプリ側の独自実装
- **サンプル画像プレビュー**: 生成画像のサムネイル表示

