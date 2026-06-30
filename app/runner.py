"""app/runner.py — 学習プロセスの起動・停止・ログ転送。

TrainState のキュー(_log_queue)にログ文字列を投入し、
GUI 側の after() ポーリング(start_log_drain)でテキストウィジェットへ反映する。
グラフデータは TrainState.push_graph_point() 経由で書き込み、
tab_monitor.py の MonitorGraph がポーリングして読み取る。

ログパース対象（musubi-tuner / accelerate 標準出力フォーマット）:
  - step 数        : tqdm の "steps: NN%|...| N/M [..., avr_loss=X]" 形式から N/M を抽出
  - epoch 数        : "epoch N/M" 表記
  - train loss     : "avr_loss=" または "loss=" の数値
  - learning rate  : "lr=" または "lr:" の数値
  - grad_norm      : "grad_norm=" の数値

  musubi-tuner には validation loss / EarlyStopping のログ出力が存在しないため、
  これらの抽出は実装していない（無編集の krea2_train_network.py / hv_train_network.py
  に val loss 計算ロジックがないことを確認済み）。
  また grad_norm についても、musubi-tuner の標準ログには accelerator.clip_grad_norm_()
  の結果が出力されないため、通常は取得できない（モニター側は「データなし」を正しい状態
  として扱う）。

  tqdm 重複行フィルタ:
  musubi-tuner（子プロセス側）は同一 step に対して update() と set_postfix() を
  別々に呼び出しており、同じカウンタ "N/M" を持つ行が2行連続で出力される
  （1行目は avr_loss なし、2行目は avr_loss あり、または逆）。
  これは子プロセス側の tqdm 呼び出し構造の問題であり GUI 側のバッファリング制御では
  解決できないため、GUI 側で「同一カウンタの直前の steps: 行を上書きする」方式で
  ログ表示の重複を抑制する。グラフ用データ（push_graph_point）には影響しない
  （同一 step の最新値で上書きされるだけなので問題ない）。
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
_AVR_LOSS_RE  = re.compile(r"\bavr_loss[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", re.I)
_LOSS_RE      = re.compile(r"(?<!avr_)\bloss[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?|nan|inf)", re.I)
_LR_RE        = re.compile(r"\blr[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", re.I)
_GRAD_NORM_RE = re.compile(r"\bgrad_norm[=:\s]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?|nan|inf)", re.I)
_EPOCH_RE     = re.compile(r"\bepoch\s*[=:]?\s*(\d+)", re.I)

# tqdm の "steps:" プレフィックス行から N/M カウンタを抽出する。
# 例: "steps:   2%|▏    | 13/544 [02:57<2:01:05, 13.68s/it, avr_loss=0.0791]"
_TQDM_STEPS_PREFIX_RE = re.compile(r"^steps:\s")
_TQDM_COUNTER_RE      = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


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
    s.training_end_time    = None
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

            # tqdm "steps:" 行の重複表示を抑制するための直前カウンタ記憶。
            # None = 直前が steps: 行ではなかった（通常ログ）。
            last_steps_counter: list[tuple[int, int] | None] = [None]

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"CMD: {' '.join(cmd)}\n\n")
                step_counter = [0]
                for raw_line in proc.stdout:
                    line = _ANSI_RE.sub("", raw_line.rstrip())
                    _enqueue_filtered(s, line, last_steps_counter)
                    lf.write(line + "\n")
                    lf.flush()
                    _try_parse_graph(s, line, step_counter)

            proc.wait()
            rc  = proc.returncode
            msg = f"[DONE] 終了コード: {rc}"
            _enqueue(s, msg)
            s.log_fn(msg)
            s.status_var.set("完了" if rc == 0 else f"エラー (rc={rc})")
            s.training_end_time = datetime.datetime.now()

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
            if on_finish:
                on_finish(rc)

        except Exception as exc:
            msg = f"[ERROR] 起動失敗: {exc}"
            _enqueue(s, msg)
            s.log_fn(msg)
            s.status_var.set("起動失敗")
            s.training_end_time = datetime.datetime.now()
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
    # NOTE: 実プロセスの終了は _worker() 側の proc.wait() 後に検知され、
    #       そこで training_end_time が確定する（停止シグナル送信時点では
    #       プロセスがまだ後処理中の可能性があるため、ここでは設定しない）。


def start_log_drain(
    widget_getter: Callable[[], list[tk.Text]],
    s: TrainState,
    root: tk.Misc,
    interval_ms: int = 200,
) -> None:
    """ログキューを定期的にドレインして Text ウィジェットへ追記する。

    1 セッションにつき 1 回だけ呼び出す（重複登録を防ぐため _log_drain_started フラグで管理）。

    キューに積まれるアイテムは (msg, replace_last) のタプル。
    replace_last が True の場合、ウィジェットの最終行を削除してから msg を挿入する
    （tqdm "steps:" 行の重複表示抑制用）。
    """
    if s._log_drain_started:
        return
    s._log_drain_started = True

    def _drain() -> None:
        widgets = widget_getter()
        try:
            while True:
                item = s._log_queue.get_nowait()
                if isinstance(item, tuple):
                    msg, replace_last = item
                else:
                    # 後方互換: 文字列のみが積まれていた場合
                    msg, replace_last = item, False

                for w in widgets:
                    w.configure(state=tk.NORMAL)
                    if replace_last:
                        _delete_last_line(w)
                    w.insert(tk.END, msg + "\n")
                    w.see(tk.END)
                    w.configure(state=tk.DISABLED)
        except Exception:
            pass
        root.after(interval_ms, _drain)

    root.after(interval_ms, _drain)


def _delete_last_line(w: tk.Text) -> None:
    """Text ウィジェットの最終行（末尾の空行を除く）を削除する。"""
    # end-1c は末尾の暗黙の改行の直前を指す。そこからさらに1行分遡って削除する。
    end_index = w.index("end-1c")
    line_start = w.index(f"{end_index} linestart")
    w.delete(line_start, "end")


# ── 内部ヘルパー ─────────────────────────────────────────────────────────────

def _enqueue(s: TrainState, msg: str) -> None:
    """ログをキューに投入する（置換なし、通常追記）。"""
    s._log_queue.put((msg, False))


def _enqueue_filtered(
    s: TrainState,
    line: str,
    last_steps_counter: list[tuple[int, int] | None],
) -> None:
    """tqdm "steps:" 行の重複を抑制してキューに投入する。

    同一カウンタ (N/M) の steps: 行が連続した場合、直前の表示行を
    上書き（置換）する。カウンタが異なる、または steps: 行でない場合は
    通常通り追記する。

    update() と set_postfix() の分離呼び出しにより、同一カウンタの行が
    avr_loss なし→ありの順で2行出力されるケースに対応する
    （子プロセス側の tqdm 呼び出し構造の問題であり、GUI 側ではこの
    フィルタでのみ抑制可能）。
    """
    if _TQDM_STEPS_PREFIX_RE.match(line):
        counter_m = _TQDM_COUNTER_RE.search(line)
        if counter_m:
            counter = (int(counter_m.group(1)), int(counter_m.group(2)))
            if last_steps_counter[0] == counter:
                # 直前と同一カウンタ → 置換
                s._log_queue.put((line, True))
                return
            last_steps_counter[0] = counter
            s._log_queue.put((line, False))
            return
        # カウンタが抽出できない steps: 行（稀） → 通常追記しつつ記憶をリセット
        last_steps_counter[0] = None
        s._log_queue.put((line, False))
        return

    # steps: 行でない通常ログ → 記憶をリセットして追記
    last_steps_counter[0] = None
    s._log_queue.put((line, False))


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
    """ログ行から avr_loss / loss / lr / grad_norm / epoch / step を抽出して
    TrainState のグラフバッファに追記する。

    step 番号は tqdm "steps:" 行の N/M カウンタの N を優先的に使用し、
    取得できなければ内部カウンタで代用する。
    avr_loss を優先的に探索し、なければ汎用 loss= にフォールバックする。
    """
    loss_m = _AVR_LOSS_RE.search(line)
    if not loss_m:
        loss_m = _LOSS_RE.search(line)
    if not loss_m:
        return

    counter_m = _TQDM_COUNTER_RE.search(line)
    if counter_m:
        step = int(counter_m.group(1))
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

    # grad_norm 抽出（TrainState 側に対応メソッドがある場合のみ記録）。
    # 無編集の musubi-tuner はこの値をログに出力しないため、通常は常に未検出となる。
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
