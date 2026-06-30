"""app/runner.py — 学習プロセスの起動・停止・ログ転送。

TrainState のキュー(_log_queue)にログ文字列を投入し、
GUI 側の after() ポーリング(start_log_drain)でテキストウィジェットへ反映する。
グラフデータは TrainState.push_graph_point() 経由で書き込み、
tab_monitor.py の MonitorGraph がポーリングして読み取る。

ログパース対象（musubi-tuner / accelerate 標準出力フォーマット）:
  - step 数        : tqdm プログレスバー由来の "N/M" または "step:N" 表記
  - epoch 数        : "epoch N/M" 表記
  - train loss     : "avr_loss=" または "loss=" の数値
  - learning rate  : "lr=" または "lr:" の数値
  - grad_norm      : "grad_norm=" の数値（accelerate の gradient clipping ログ）

  musubi-tuner には validation loss / EarlyStopping のログ出力が存在しないため、
  これらの抽出は実装していない（無編集の krea2_train_network.py / hv_train_network.py
  に val loss 計算ロジックがないことを確認済み）。
"""
from __future__ import annotations

import datetime
import os
import re
import signal
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from typing import Callable

from .state import TrainState

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# avr_loss を優先的に探し、なければ汎用 loss= にフォールバック
_AVR_LOSS_RE = re.compile(r"\bavr_loss[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", re.I)
_LOSS_RE     = re.compile(r"(?<!avr_)\bloss[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?|nan|inf)", re.I)
_LR_RE       = re.compile(r"\blr[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", re.I)
_GRAD_NORM_RE = re.compile(r"\bgrad_norm[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?|nan|inf)", re.I)
_STEP_RE     = re.compile(r"\b(?:step|iter)[=:\s]+(\d+)", re.I)
_EPOCH_RE    = re.compile(r"\bepoch\s*[=:]?\s*(\d+)", re.I)


def launch_training(
    s: TrainState,
    cmd: list[str],
    on_finish: Callable[[int], None] | None = None,
) -> None:
    """学習コマンドをバックグラウンドスレッドで起動する。"""
    if s.is_running():
        s.log_fn("[WARN] 既に学習が実行中です。")
        return

    log_dir = s.paths.log / "krea2_train"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{ts}.txt"

    s.log_fn(f"[START] ログ: {log_path}")
    s.status_var.set("実行中…")
    s.training_start_time = datetime.datetime.now()
    _enqueue(s, f"[CMD] {' '.join(cmd)}")

    def _worker() -> None:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(s.paths.musubi),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=_creation_flags(),
            )
            s._proc = proc

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"CMD: {' '.join(cmd)}\n\n")
                step_counter = [0]
                for raw_line in proc.stdout:
                    line = _ANSI_RE.sub("", raw_line.rstrip())
                    _enqueue(s, line)
                    lf.write(line + "\n")
                    lf.flush()
                    _try_parse_graph(s, line, step_counter)

            proc.wait()
            rc  = proc.returncode
            msg = f"[DONE] 終了コード: {rc}"
            _enqueue(s, msg)
            s.log_fn(msg)
            s.status_var.set("完了" if rc == 0 else f"エラー (rc={rc})")

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
            if on_finish:
                on_finish(rc)

        except Exception as exc:
            msg = f"[ERROR] 起動失敗: {exc}"
            _enqueue(s, msg)
            s.log_fn(msg)
            s.status_var.set("起動失敗")
        finally:
            s._proc = None

    threading.Thread(target=_worker, daemon=True).start()


def stop_training(s: TrainState) -> None:
    """実行中の学習プロセスに停止シグナルを送る。"""
    if not s.is_running():
        s.log_fn("[WARN] 実行中のプロセスがありません。")
        return
    try:
        os.kill(s._proc.pid, signal.CTRL_BREAK_EVENT)
    except (AttributeError, OSError):
        s._proc.terminate()
    s.status_var.set("停止要求済み")
    s.log_fn("[STOP] 停止シグナルを送信しました。")


def start_log_drain(
    widget_getter: Callable[[], list[tk.Text]],
    s: TrainState,
    root: tk.Misc,
    interval_ms: int = 200,
) -> None:
    """ログキューを定期的にドレインして Text ウィジェットへ追記する。

    1 セッションにつき 1 回だけ呼び出す（重複登録を防ぐため _log_drain_started フラグで管理）。
    """
    if s._log_drain_started:
        return
    s._log_drain_started = True

    def _drain() -> None:
        widgets = widget_getter()
        try:
            while True:
                msg = s._log_queue.get_nowait()
                for w in widgets:
                    w.configure(state=tk.NORMAL)
                    w.insert(tk.END, msg + "\n")
                    w.see(tk.END)
                    w.configure(state=tk.DISABLED)
        except Exception:
            pass
        root.after(interval_ms, _drain)

    root.after(interval_ms, _drain)


# ── 内部ヘルパー ─────────────────────────────────────────────────────────────

def _enqueue(s: TrainState, msg: str) -> None:
    """ログをキューに投入する。"""
    s._log_queue.put(msg)


def _parse_float_or_special(text: str) -> float:
    """'nan' / 'inf' 文字列を float('nan') / float('inf') に変換する。"""
    low = text.strip().lower()
    if low == "nan":
        return float("nan")
    if low == "inf":
        return float("inf")
    return float(text)


def _try_parse_graph(
    s: TrainState,
    line: str,
    step_counter: list[int],
) -> None:
    """ログ行から avr_loss / loss / lr / grad_norm / epoch を抽出して
    TrainState のグラフバッファに追記する。

    avr_loss を優先的に探索し、なければ汎用 loss= にフォールバックする。
    grad_norm が見つかった場合は s.push_grad_norm_point() があれば併せて記録する
    （TrainState 側の対応状況に応じて欠落しても処理を継続する）。
    step 番号はログに含まれていれば使用し、なければ内部カウンタで代用する。
    """
    loss_m = _AVR_LOSS_RE.search(line)
    if not loss_m:
        loss_m = _LOSS_RE.search(line)
    if not loss_m:
        return

    step_m = _STEP_RE.search(line)
    if step_m:
        step = int(step_m.group(1))
    else:
        step_counter[0] += 1
        step = step_counter[0]

    lr_m = _LR_RE.search(line)
    lr   = float(lr_m.group(1)) if lr_m else 0.0

    try:
        loss = _parse_float_or_special(loss_m.group(1))
    except ValueError:
        return

    s.push_graph_point(step, loss, lr)

    # epoch 抽出（あれば最新値を更新）
    epoch_m = _EPOCH_RE.search(line)
    if epoch_m and hasattr(s, "push_epoch_point"):
        try:
            s.push_epoch_point(int(epoch_m.group(1)))
        except (ValueError, AttributeError):
            pass

    # grad_norm 抽出（TrainState 側に対応メソッドがある場合のみ記録）
    gn_m = _GRAD_NORM_RE.search(line)
    if gn_m and hasattr(s, "push_grad_norm_point"):
        try:
            grad_norm = _parse_float_or_special(gn_m.group(1))
            s.push_grad_norm_point(step, grad_norm)
        except (ValueError, AttributeError):
            pass


def _creation_flags() -> int:
    """Windows: 新プロセスグループで起動するフラグを返す。非 Windows では 0。"""
    try:
        return subprocess.CREATE_NEW_PROCESS_GROUP
    except AttributeError:
        return 0
