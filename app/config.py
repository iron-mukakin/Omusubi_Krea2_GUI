"""app/config.py — アプリケーションパス管理。

GUI ルート直下に musubi-tuner が git clone される前提で
全ディレクトリパスを一元管理する。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """アプリケーション全体のパスを保持する不変オブジェクト。"""

    root: Path          # GUI アプリルート（このファイルの2階層上）
    app: Path           # app/
    musubi: Path        # musubi-tuner/
    musubi_src: Path    # musubi-tuner/src
    musubi_venv_py: Path  # musubi-tuner/.venv/Scripts/python.exe
    preset: Path        # preset/
    log: Path           # log/
    output: Path        # output/

    @classmethod
    def from_root(cls, root: Path | None = None) -> "AppPaths":
        """root を基点にパスを解決する。root=None の場合は自動検出。"""
        base = (root or Path(__file__).resolve().parents[1]).resolve()
        musubi = base / "musubi-tuner"
        return cls(
            root=base,
            app=base / "app",
            musubi=musubi,
            musubi_src=musubi / "src",
            musubi_venv_py=musubi / ".venv" / "Scripts" / "python.exe",
            preset=base / "preset",
            log=base / "log",
            output=base / "output",
        )

    def ensure_dirs(self) -> None:
        """必要なディレクトリを作成する。"""
        for directory in (
            self.app,
            self.preset,
            self.log,
            self.output,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def krea2_train_script(self) -> Path:
        """krea2_train_network.py への絶対パス。"""
        return self.musubi_src / "musubi_tuner" / "krea2_train_network.py"

    @property
    def krea2_cache_latents_script(self) -> Path:
        return self.musubi_src / "musubi_tuner" / "krea2_cache_latents.py"

    @property
    def krea2_cache_te_script(self) -> Path:
        return self.musubi_src / "musubi_tuner" / "krea2_cache_text_encoder_outputs.py"

    def validate_musubi(self) -> str | None:
        """musubi-tuner の存在確認。問題があればエラーメッセージを返す。"""
        if not self.musubi.exists():
            return f"musubi-tuner が見つかりません: {self.musubi}\nsetup.bat を実行してください。"
        if not self.krea2_train_script.exists():
            return f"学習スクリプトが見つかりません: {self.krea2_train_script}"
        if not self.musubi_venv_py.exists():
            return f"仮想環境が見つかりません: {self.musubi_venv_py}\nsetup.bat を実行してください。"
        return None
