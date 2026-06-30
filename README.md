# Omusubi_Krea2_gui

## はじめに
本アプリは仮リリースとしています。[kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)を使用するGUIアプリです。本アプリの他にmusubi-tunerを使用します。


## 使用方法
本プロジェクトをgit cloneした後にプロジェクトフォルダ内で[kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)をgit cloneして下さい。

本アプリのsetup.batを起動するとmusubi-tunerのプロジェクト内に仮想環境を作ります。本アプリもmusubi-tunterの仮想環境を使用して動きます。フォークでなく連携アプリ方式なのでmusubi-tuner単体稼働も可能なはずです。python3.12から優先インストールされます。

仮想環境が出来たらstart.batでアプリを立ち上げます。lora学習の前準備であるlatentキャッシュの作成もGUI上で行う事ができます。パイプライン処理で無いので学習前にキャッシュ作成を手動で行ってください。


## 留意事項
モニターグラフ、サンプル生成は開発中です。lora学習のstepが進む所までは確認済みです。CAMEなどの一部オプティマイザーはライブラリの追加インストールが必要になる事があります。

詳細最適化で--compile(torch compile)を有効にするにはtriton-windowsとMSVC（Visual Studio 2022のC++ビルドツール）が必要です。

Attentionモードでflashを有効にする場合はflashattentionをインストールする必要があります。現状のsetup.batはその他オプションには非対応なので手動導入が必要になります。

## フォルダ・ファイル構成

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

## PyTorchのバージョンについて

`--attn_mode`に`torch`を指定する場合、2.5.1以降のPyTorchを使用してください（それより前のバージョンでは生成される動画が真っ黒になるようです）。

古いバージョンを使う場合、xformersやSageAttentionを使用してください。


## 免責事項

このリポジトリは非公式であり、サポートされているアーキテクチャの公式リポジトリとは関係ありません。また、このリポジトリは開発中で、実験的なものです。テストおよびフィードバックを歓迎しますが、以下の点にご注意ください：

- 実際の稼働環境での動作を意図したものではありません
- 機能やAPIは予告なく変更されることがあります
- いくつもの機能が未検証です
- 動画学習機能はまだ開発中です


## ライセンス

Krea2のモデルライセンスは[Krea2](https://www.krea.ai/krea-2-licensing)に従います。

`hunyuan_model`ディレクトリ以下のコードは、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)のコードを一部改変して使用しているため、そちらのライセンスに従います。

`wan`ディレクトリ以下のコードは、[Wan2.1](https://github.com/Wan-Video/Wan2.1)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

`frame_pack`ディレクトリ以下のコードは、[frame_pack](https://github.com/lllyasviel/FramePack)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

他のコードはApache License 2.0に従います。一部Diffusersのコードをコピー、改変して使用しています。
