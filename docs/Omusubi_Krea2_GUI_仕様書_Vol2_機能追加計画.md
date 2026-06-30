# Omusubi_Krea2_GUI — 機能追加仕様書 Vol.2
**今後の実装計画 / AIエージェント向け指示書**
バージョン: 1.0.0（2026-06-28 策定）

---

## 凡例

| 優先度 | 記号 | 説明 |
|---|---|---|
| 高 | 🔴 | 学習精度・利便性に直結。早期実装推奨 |
| 中 | 🟡 | 品質向上。実装順序は柔軟 |
| 低 | 🟢 | 拡張・将来対応 |

---

## Part 1 — Krea2 LoRA 学習タブの強化

---

### F-001 🔴 階層学習（Layer-wise LR）

**目的**: ブロックごとに異なる学習率を設定し、学習の精度と効率を向上させる。

**背景**:
Krea2 の DiT は 28 ブロック（`blocks.0`〜`blocks.27`）＋ txtfusion で構成される。
現状は `--network_args exclude_patterns / include_patterns` で対象ブロックを絞るだけ。
各ブロックに倍率スケールを設定する `lr_scheduler_kwargs` 方式を GUI から設定できるようにする。

**実装場所**: `app/` 以下に新規ファイル `tab_layer_lr.py` を追加。
`tab_lora.py` の `setting_specs` に `("階層学習", build_layer_lr_tab)` を追加。

**UI 設計**:
```
[階層学習] タブ
├── 有効チェックボックス「階層学習を有効にする」
├── LabelFrame「ブロック別 LR 倍率」
│   ├── スライダー × 28（blocks.0〜27）  各 0.0〜2.0
│   ├── txtfusion スライダー              0.0〜2.0
│   └── [全リセット] [均等設定] ボタン
├── LabelFrame「プリセット」
│   ├── 「入力層のみ高LR」「出力層のみ」「リニア減衰」等の定型
└── [プレビュー] → network_args 文字列をポップアップ表示
```

**CLI への反映**:
```python
# network_args に block_lr_scale を追加
--network_args "block_lr_scale={'blocks.0':0.1,'blocks.27':1.0,...}"
```

**state.py への追加変数**:
```python
self.layer_lr_enabled = tk.BooleanVar(value=False)
self.layer_lr_scales: dict[str, tk.DoubleVar]  # key="blocks.N", "txtfusion"
```

**preset_manager.py 対応**:
`collect_preset` / `apply_preset` に `layer_lr_enabled`, `layer_lr_scales` を追加。

---

### F-002 🔴 Entry Stopping（早期停止）

**目的**: 検証 loss が改善しなくなった時点で学習を自動停止させる。
過学習防止とGPU時間節約。

**実装場所**: `app/tab_train.py` に追加セクション。
GUIアプリ側の独自実装（musubi-tuner 本体には機能なし）。

**仕組み**:
1. `runner.py` の `_try_parse_graph()` で loss をリアルタイム取得
2. `EntryStoppingMonitor` クラスが N エポック連続で改善なしを検出
3. `stop_training(s)` を自動呼び出し

**UI 設計**:
```
[学習設定] タブ内 LabelFrame「早期停止」
├── チェックボックス「早期停止を有効にする」
├── patience（改善なし継続エポック数）: Spinbox 1〜50 デフォルト5
├── min_delta（改善とみなす最小変化量）: Entry デフォルト0.001
└── ステータス表示「現在 3/5 エポック連続で未改善」
```

**新規ファイル**: `app/early_stopping.py`

```python
class EntryStoppingMonitor:
    def __init__(self, patience: int, min_delta: float): ...
    def record(self, epoch: int, loss: float) -> bool:
        """True を返したら停止シグナルを発行すべき。"""
    def reset(self) -> None: ...
    def status_message(self) -> str: ...
```

**state.py への追加**:
```python
self.early_stop_enabled = tk.BooleanVar(value=False)
self.early_stop_patience = tk.IntVar(value=5)
self.early_stop_min_delta = tk.DoubleVar(value=0.001)
```

---

### F-003 🟡 サンプル生成の強化

**目的**: 学習中サンプルの管理とギャラリー表示の改善。

**現状**: `tab_sample.py` にギャラリー（最新10枚表示）あり。PIL が必要。

**追加機能**:

#### F-003a サンプル画像ビューア
- クリックで拡大表示（`tk.Toplevel` に `Canvas` で表示）
- エポック番号をファイル名から自動解析して表示

#### F-003b サンプル生成パラメータ拡張
```
現状: width, height, steps, guidance_scale, seed
追加: cfg_scale_negative（ネガティブ CFG）, sampler（euler等）
```

#### F-003c ギャラリー自動更新の最適化
- 現状: `after(3000, ...)` で全再スキャン
- 改善: `watchdog` または `os.stat` での差分検出に変更

---

### F-004 🟡 モニターグラフの強化

**目的**: 学習進捗の可視化精度向上。

**現状**: `tab_monitor.py` に loss / lr グラフ（matplotlib）。
ログパーサーが `runner.py` 内で step / loss / lr を正規表現で抽出。

**追加機能**:

#### F-004a ログパーサーの精度向上

現状の正規表現:
```python
_LOSS_RE  = re.compile(r"\bloss[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", re.I)
_LR_RE    = re.compile(r"\blr[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", re.I)
_STEP_RE  = re.compile(r"\b(?:step|iter|epoch)[=:\s]+(\d+)", re.I)
```

musubi-tuner の実際のログフォーマットに合わせて調整が必要。
→ 学習実行後に実ログを取得して正規表現をチューニングする。

#### F-004b グラフ追加項目
- epoch 境界の垂直線
- 移動平均線（ウィンドウサイズ調整可能）
- グラフの PNG エクスポートボタン

**実装場所**: `tab_monitor.py` の `MonitorGraph` クラスを拡張。

---

### F-005 🟡 学習再開（Resume）

**目的**: 途中で停止した学習を再開できるようにする。

**CLI 引数**: `--resume <state_file>` （musubi-tuner 対応済み）

**UI 設計**:
```
[学習設定] タブ内 LabelFrame「学習再開」
├── resume パス: Entry + Browse (.json/.safetensors)
└── チェック「save_state を有効にする」
```

**state.py への追加**:
```python
self.resume_path = tk.StringVar()
self.save_state = tk.BooleanVar(value=False)
self.save_state_on_train_end = tk.BooleanVar(value=True)
```

**cmd_builder.py への追加**:
```python
_append_optional_str(cmd, "--resume", s.resume_path.get())
if s.save_state.get():
    cmd.append("--save_state")
if s.save_state_on_train_end.get():
    cmd.append("--save_state_on_train_end")
```

---

### F-006 🟡 学習ログの詳細表示

**目的**: 実行タブのログを改善して学習状況を把握しやすくする。

**追加機能**:
- 学習速度（it/s）の表示
- 残り時間の推定（ETA）
- epoch / step の進捗バー（`ttk.Progressbar`）
- ログのフィルタ（ERROR のみ / WARNING 以上 / 全表示）

**実装場所**: `run_panel.py` を拡張。
`state.py` に `total_steps` / `current_step` 変数を追加。

---

## Part 2 — データセット管理の強化

---

### F-007 🔴 キャッシュ生成の改善

**目的**: キャッシュ生成の信頼性向上と使いやすさの改善。

#### F-007a キャッシュ状態の可視化

**実装場所**: `tab_cache.py` に「キャッシュ状態確認」セクションを追加。

```
[キャッシュ生成] タブ
├── LabelFrame「キャッシュ状態」
│   ├── [確認] ボタン → image_dir をスキャン
│   ├── 画像数: 34
│   ├── Latent キャッシュ: 34/34 ✓
│   └── TE キャッシュ: 0/34 ✗ → 要生成
```

**実装方針**: `cache_directory` 内の `.npz` ファイル数を `image_dir` 内の画像数と比較。

#### F-007b キャッシュ再生成ボタン
- `--skip_existing` なしの強制再生成オプション

---

### F-008 🟡 データセットタブの拡張

**目的**: musubi-tuner スキーマの全フィールドを GUI から設定できるようにする。

**現在サポート外のフィールド**（musubi-tuner に存在するが GUI にない）:
```toml
[[datasets]]
image_jsonl_file = ""        # jsonl 形式の画像リスト
control_directory = ""       # ControlNet 用
multiple_target = false      # マルチターゲット学習
debug_dataset = false        # デバッグ出力
```

**追加 UI**: `tab_dataset.py` のエントリカードに「詳細設定（折り畳み）」セクションを追加。

---

## Part 3 — 新機能タブの追加

---

### F-009 🟡 マージタブ

**目的**: LoRA モデルのマージ・重み操作を GUI から実施する。

**実装場所**: `app/tab_merge.py`（新規）
`main.py` のグローバルノートブックに `[マージ]` タブとして追加。

**UI 設計**:
```
[マージ] タブ
├── LabelFrame「マージ方式」
│   ├── ラジオ: LoRA 合成 / 加重平均 / DARE
│   └── alpha スライダー
├── LabelFrame「入力モデル」
│   ├── Base モデル (DiT RAW): Browse
│   ├── LoRA A: Browse + 倍率
│   └── LoRA B: Browse + 倍率（オプション）
├── LabelFrame「出力設定」
│   ├── 出力フォルダ / ファイル名
│   └── save_precision
└── [▶ マージ実行] [コマンドプレビュー]
```

**依存**: musubi-tuner に `merge_lora` 相当スクリプトがある場合はそれを使用。
なければ `safetensors` + `torch` でカスタム実装。

---

### F-010 🟡 推論タブ（Generate）

**目的**: 学習済み LoRA を使った画像生成を GUI から実施する。

**実装場所**: `app/tab_generate.py`（新規）

**依存スクリプト**: `krea2_generate_image.py`（アップロード済みファイルに存在）

**UI 設計**:
```
[生成] タブ
├── モデル設定
│   ├── DiT (RAW/Turbo): Browse
│   ├── VAE: Browse
│   └── Text Encoder: Browse
├── LoRA 設定
│   ├── LoRA ファイル: Browse
│   └── 倍率: 0.0〜2.0
├── 生成設定
│   ├── プロンプト / ネガティブプロンプト
│   ├── width / height / steps / guidance_scale / seed
│   └── batch_count
├── [▶ 生成] ボタン
└── プレビューギャラリー（生成結果）
```

**実装方針**:
```python
# cmd_builder 相当の関数を tab_generate.py 内に実装
cmd = [python, str(paths.musubi_src / "musubi_tuner" / "krea2_generate_image.py"),
       "--dit", ..., "--vae", ..., "--text_encoder", ...,
       "--lora_weights", ..., "--lora_multiplier", ...,
       "--prompt", ..., "--output_dir", ...]
```

---

### F-011 🟢 ログ解析タブ

**目的**: 過去の学習ログを読み込んでグラフ表示・比較する。

**実装場所**: `app/tab_log_analysis.py`（新規）

**UI 設計**:
```
[ログ解析] タブ
├── ログファイル選択（複数選択可）
├── グラフ表示（matplotlib）
│   ├── loss 比較（複数ラン）
│   └── lr 変化
├── 統計情報（最小 loss / 最終 loss / 総ステップ数）
└── [PNG エクスポート]
```

---

### F-012 🟢 設定タブ

**目的**: アプリ全体の設定を管理する。

**実装場所**: `app/tab_settings.py`（新規）
グローバルノートブックに `[設定]` タブとして追加。

**設定項目**:
```
[設定] タブ
├── LabelFrame「パス設定」
│   ├── musubi-tuner パス: Browse（デフォルト ./musubi-tuner）
│   └── Python / venv パス: 自動検出 or Browse
├── LabelFrame「GUI 設定」
│   ├── テーマ: [vista / clam / alt / default]
│   ├── フォントサイズ
│   └── ウィンドウサイズ記憶
├── LabelFrame「musubi-tuner」
│   ├── [git pull] ボタン（アップデート）
│   └── バージョン表示
└── [保存] → app_config.json に書き出し
```

**新規ファイル**: `app/app_config.py`

```python
@dataclass
class AppConfig:
    musubi_dir: str = "musubi-tuner"
    theme: str = "vista"
    font_size: int = 9
    window_size: str = "1200x820"

    @classmethod
    def load(cls, path: Path) -> "AppConfig": ...
    def save(self, path: Path) -> None: ...
```

---

## Part 4 — コード品質・アーキテクチャ改善

---

### F-013 🟡 ウィジェット命名規則の関数名警告リスト対応

**現状**: `widgets.py` の関数名に命名規則（動詞+対象）が一部不徹底。

**対象関数の改名候補**:

| 現在 | 推奨 | 理由 |
|---|---|---|
| `labeled_frame` | `attach_labeled_frame` | 警告リスト外だが動詞+対象に統一 |
| `attach_log_widget` | OK | 適切 |
| `browse_file_row` | `place_browse_file_row` | より明確 |

→ 改名する場合は全タブファイルの呼び出し箇所を一括修正すること。
`apply_fix_*.py` パターンで対応可能。

---

### F-014 🟡 preset_manager のスキーマバージョン管理

**問題**: プリセット JSON のスキーマが変わるたびに後方互換問題が発生している。

**解決策**: JSON に `schema_version` フィールドを追加し、
`apply_preset` でバージョン差異を吸収するマイグレーション関数を実装する。

**実装場所**: `preset_manager.py`

```python
CURRENT_SCHEMA_VERSION = 2

def _migrate(data: dict) -> dict:
    """旧バージョンの JSON を現行スキーマに変換する。"""
    version = data.get("schema_version", 1)
    if version < 2:
        # v1 → v2: dataset_config → dataset_mode + dataset_entries
        data["dataset_mode"] = "toml" if data.get("dataset_config") else "gui"
        data.setdefault("dataset_entries", [_default_entry()])
        data.setdefault("general_resolution", 1024)
        data.pop("dataset_config", None)
    data["schema_version"] = CURRENT_SCHEMA_VERSION
    return data
```

---

### F-015 🟡 pack/grid レイアウト自動チェック

**問題**: タブ追加時に `pack` と `grid` の混在エラーが繰り返し発生している。

**解決策**: CI 相当の静的チェックスクリプトを追加。

**新規ファイル**: `check_layout.py`（GUIルート直下）

```python
"""pack/grid 混在をチェックするスクリプト。"""
import ast, pathlib, sys, collections

def check_file(path: pathlib.Path) -> list[str]:
    """同一変数名に grid と pack が混在していれば警告を返す。"""
    ...

if __name__ == "__main__":
    errors = []
    for f in pathlib.Path("app").glob("tab_*.py"):
        errors.extend(check_file(f))
    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    print("OK: no pack/grid conflicts")
```

---

### F-016 🟢 型アノテーションの完全化

**現状**: 一部の関数に型アノテーションが不足している（`log_fn` の型など）。

**追加すべきアノテーション**:
```python
# main.py
def _make_log_fn(...) -> Callable[[str], None]: ...

# tab_lora.py
def build_lora_tab(..., log_fn: Callable[[str], None]) -> TrainState: ...

# runner.py
def launch_training(..., on_finish: Callable[[int], None] | None = None) -> None: ...
```

---

### F-017 🟢 テストスイートの整備

**目的**: 回帰テストを自動化して品質を保証する。

**テスト対象**:
1. `dataset_config.py` の toml 生成（禁止キー検査含む）✅ 既に手動テストあり
2. `cmd_builder.py` のコマンド生成（引数の正しさ）
3. `preset_manager.py` のラウンドトリップ
4. `_krea2_launcher.py` の spawn 安全性 ✅ 既に手動テストあり
5. `tab_dataset.py` の `resolve_dataset_config`

**新規ファイル**: `tests/test_dataset_config.py`, `tests/test_cmd_builder.py`

---

## Part 5 — 将来モデル対応

---

### F-018 🟢 他アーキテクチャの LoRA 学習タブ

**目的**: musubi-tuner がサポートする他モデルにも対応する。

**候補アーキテクチャ**:
- HunyuanVideo（動画 LoRA）
- Wan2.1（動画 LoRA）
- FLUX.1 Kontext
- HiDream-o1

**設計方針**:
- グローバルノートブックに `[HunyuanVideo LoRA 学習]` 等のタブを追加
- 各アーキテクチャ固有の `tab_<arch>_lora.py` を `app/` に追加
- `TrainState` の共通部分は基底クラス化を検討
- `cmd_builder.py` をアーキテクチャごとに分割（`cmd_builder_krea2.py` 等）

**実装優先順序**: HunyuanVideo → Wan2.1 → FLUX.1 Kontext

---

## Part 6 — 実装指示テンプレート

AIエージェントへの指示に使用するテンプレート。

---

### 新規タブ追加の手順（テンプレート）

```
【指示】app/ に新規タブ tab_XXXX.py を追加してください。

【仕様書参照】
- Vol.1 § 4.2 TrainState（変数追加が必要な場合）
- Vol.1 § 4.9 / 4.10 タブ実装パターン
- Vol.1 § 8 既知制約

【必須要件】
1. レイアウト: parent への配置を grid または pack に統一（混在禁止）
2. 関数命名: 動詞+対象形式（warnings list: run, process, handle, manager, helper,
   save, load, update, create, build, execute, init, start, stop, open）
3. 型アノテーション: 全引数・戻り値に付与
4. state.py への変数追加:  TrainState.__init__ の対応グループに追加
5. preset_manager.py: collect_preset / apply_preset に追加
6. tab_lora.py: setting_specs リストに追加

【ファイルへの反映先】
- app/tab_XXXX.py  （新規）
- app/state.py     （変数追加）
- app/tab_lora.py  （setting_specs に追加）
- app/preset_manager.py  （collect/apply に追加）
- preset/krea2/default_krea2.json  （デフォルト値を追加）

【パッチ方式】
修正は apply_fix_XXXX.py で適用（Python 純正、patch コマンド不要）
```

---

### CLI 引数追加の手順（テンプレート）

```
【指示】--new_flag を krea2_train_network.py に渡すよう対応してください。

【確認事項】
1. krea2_train_network.py のヘルプ出力に --new_flag が存在するか確認
   （存在しない引数を渡すと "unrecognized arguments" エラー）
2. Krea2 固有の排他制約がないか確認

【修正ファイル】
- app/state.py: new_flag = tk.BooleanVar(value=False) を追加
- app/cmd_builder.py: bool_flags リストに (s.new_flag, "--new_flag") を追加
- app/tab_advanced.py または対応タブ: Checkbutton を追加
- app/preset_manager.py: collect_preset / apply_preset に追加
- preset/krea2/default_krea2.json: "new_flag": false を追加
```

---

## 実装ロードマップ

```
Phase 1（高優先・学習品質）
├── F-001 階層学習（Layer-wise LR）
├── F-002 Entry Stopping（早期停止）
└── F-007 キャッシュ状態可視化

Phase 2（中優先・利便性）
├── F-005 学習再開（Resume）
├── F-006 学習ログ詳細表示
├── F-009 マージタブ
├── F-010 推論タブ（Generate）
└── F-014 プリセットスキーマバージョン管理

Phase 3（品質・拡張）
├── F-011 ログ解析タブ
├── F-012 設定タブ
├── F-015 pack/grid 自動チェック
├── F-017 テストスイート
└── F-003/F-004 サンプル・モニター強化

Phase 4（マルチアーキテクチャ）
└── F-018 HunyuanVideo / Wan2.1 / FLUX.1 対応
```

---

## 付録 A — AIエージェントへの共通注意事項

```
1. 絶対パスのコード埋め込み禁止
   → AppPaths 経由で全パスを解決する

2. 関数命名の警告リスト
   禁止ワード: run, process, handle, manager, helper, save, load,
              update, create, build, execute, init, start, stop, open
   → 動詞+対象形式: fetch_xxx, write_xxx, attach_xxx, collect_xxx 等

3. レイアウト混在禁止
   → 同一 parent に pack と grid を混在させない
   → labeled_frame() は parent に pack するため、
     その戻り値の子は grid を使う

4. TrainState 変数の追加場所
   → state.py の __init__ 内、対応グループのコメント配下に追記
   → private 変数（_で始まる）は __init__ の最初のブロック

5. preset_manager.py の同期
   → state.py に変数を追加したら必ず
     collect_preset と apply_preset の両方に追加する

6. Windows spawn 対応
   → DataLoader を使うスクリプトを起動する場合は
     launcher に freeze_support() + if __name__ == '__main__' が必要

7. musubi-tuner スキーマ厳守
   → dataset_config.toml に [[datasets.subsets]] は書かない
   → shuffle_caption, keep_tokens, flip_aug, min_bucket_reso,
     max_bucket_reso は musubi-tuner のスキーマに存在しない

8. コード行数の制約
   → 1ファイルあたり 350 行以下を目標とする
   → 超える場合はサブモジュールへの分割を検討
```

---

## 付録 B — 現行ファイル行数一覧（参考）

| ファイル | 行数 | 役割 |
|---|---|---|
| `__init__.py` | 1 | パッケージ宣言 |
| `cmd_builder.py` | 271 | コマンド生成 |
| `config.py` | 72 | パス管理 |
| `dataset_config.py` | 141 | toml 生成 |
| `main.py` | 125 | エントリポイント |
| `preset_manager.py` | 232 | プリセット管理 |
| `run_panel.py` | 102 | 実行パネル |
| `runner.py` | 184 | プロセス管理 |
| `state.py` | 219 | 状態管理 |
| `tab_advanced.py` | 205 | 詳細・最適化タブ |
| `tab_cache.py` | 248 | キャッシュ生成タブ |
| `tab_dataset.py` | 345 | データセットタブ |
| `tab_lora.py` | 71 | LoRAタブオーケストレーター |
| `tab_model.py` | 48 | モデルタブ |
| `tab_monitor.py` | 143 | モニタータブ |
| `tab_network.py` | 87 | ネットワークタブ |
| `tab_preset.py` | 123 | プリセットタブ |
| `tab_sample.py` | 193 | サンプル生成タブ |
| `tab_train.py` | 88 | 学習設定タブ |
| `widgets.py` | 175 | 共通ウィジェット |
| **合計** | **3073** | 20 ファイル |
