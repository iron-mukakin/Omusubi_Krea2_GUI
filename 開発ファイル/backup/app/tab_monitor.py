"""app/tab_monitor.py — [モニター] タブ UI ビルダー + MonitorGraph クラス。

ログキューの競合を避けるため、グラフデータは TrainState._graph_points /
_grad_norm_points（runner.py の push_*() から書き込まれるスレッドセーフバッファ）
経由で受け取る。matplotlib が利用不可の場合はフォールバックラベルを表示する。

【重要】無編集の musubi-tuner（krea2_train_network.py / hv_train_network.py）には
validation loss の計算ロジックおよび EarlyStopping のログ出力が存在しないことを
ソースコード精査で確認済み。そのため本実装では以下を扱わない:
  - Val Loss グラフ・パラメーター表示
  - ΔLoss（Train/Val 差分）
  - EarlyStopping パネル
扱うのは train avr_loss / lr / grad_norm の3系列と、epoch・step・経過時間・
推定残り時間、および grad_norm 急増 / Loss NaN の単純な自動診断のみ。
"""
from __future__ import annotations

import datetime
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import TrainState


def build_monitor_tab(parent: ttk.Frame, s: "TrainState") -> None:
    """モニタータブの UI を parent に構築する。"""
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)

    graph_frame = ttk.Frame(parent)
    graph_frame.grid(row=0, column=0, sticky=tk.NSEW)

    try:
        graph = MonitorGraph(graph_frame, s)
        s._monitor_graph = graph          # type: ignore[attr-defined]
    except Exception as exc:
        ttk.Label(
            graph_frame,
            text=(
                f"グラフ初期化エラー: {exc}\n"
                "matplotlib をインストールしてください:\n"
                "  pip install matplotlib"
            ),
            foreground="#EF4444",
            justify=tk.LEFT,
        ).pack(padx=16, pady=24, anchor=tk.W)


class MonitorGraph:
    """TrainState._graph_points / _grad_norm_points をポーリングしてグラフを更新する。

    runner.py の push_graph_point() / push_grad_norm_point() / push_epoch_point() が
    ログ行をパースして TrainState 側のバッファに追記し、このクラスは after() ポーリング
    でバッファを読み取って再描画・パネル更新する。排他制御は各 *_lock で保護する。
    """

    _POLL_MS: int = 600
    # grad_norm 急増判定: 直近平均の何倍を「急増」とみなすか
    _GRAD_SPIKE_RATIO: float = 3.0
    # 急増判定に使う直近サンプル数（最新値を除く）
    _GRAD_RECENT_WINDOW: int = 10

    def __init__(self, parent: tk.Widget, s: "TrainState") -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self._s      = s
        self._parent = parent

        # ── 上段: グラフ（loss / lr / grad_norm の3面）─────────────
        fig = Figure(figsize=(11, 3.4), dpi=96, tight_layout=True)
        self._ax_loss = fig.add_subplot(1, 3, 1)
        self._ax_lr   = fig.add_subplot(1, 3, 2)
        self._ax_gn   = fig.add_subplot(1, 3, 3)
        self._configure_axes()

        self._canvas = FigureCanvasTkAgg(fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(parent)
            toolbar_frame.pack(fill=tk.X)
            NavigationToolbar2Tk(self._canvas, toolbar_frame)
        except Exception:
            pass

        # ── 中段: 操作ボタン行 ───────────────────────────────────
        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Button(btn_row, text="グラフリセット",
                   command=self._reset).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="学習停止",
                   command=self._stop_training).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(btn_row,
                  text="Train Loss / LR / grad_norm を学習ログから自動抽出します"
                       "（musubi-tuner には Val Loss 出力がないため非対応）",
                  foreground="#64748B").pack(side=tk.LEFT, padx=8)

        # ── 下段: パラメーターパネル + 時間パネル + 診断ログ ───────
        bottom = ttk.Frame(parent)
        bottom.pack(fill=tk.X, padx=4, pady=(4, 4))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        self._param_frame = ttk.LabelFrame(bottom, text="学習パラメーター")
        self._param_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        self._param_labels: dict[str, ttk.Label] = {}
        self._build_param_panel()

        self._time_frame = ttk.LabelFrame(bottom, text="時間")
        self._time_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0))
        self._time_labels: dict[str, ttk.Label] = {}
        self._build_time_panel()

        self._diag_text = tk.Text(parent, height=3, state=tk.DISABLED,
                                  foreground="#B45309", wrap=tk.WORD)
        self._diag_text.pack(fill=tk.X, padx=4, pady=(0, 4))

        # 診断の重複出力防止用フラグ
        self._diag_grad_spike_active = False
        self._diag_loss_nan_active   = False

        self._start_polling()

    # ── パネル構築 ───────────────────────────────────────────────

    def _build_param_panel(self) -> None:
        rows = [
            ("epoch",      "epoch"),
            ("step",       "step"),
            ("lr",         "LR"),
            ("train_loss", "Train Loss"),
            ("grad_norm",  "grad norm"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(self._param_frame, text=label, width=14, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, padx=(6, 2), pady=2)
            val_lbl = ttk.Label(self._param_frame, text="—", anchor=tk.W)
            val_lbl.grid(row=i, column=1, sticky=tk.W, padx=(0, 6), pady=2)
            self._param_labels[key] = val_lbl

    def _build_time_panel(self) -> None:
        rows = [
            ("start",        "開始"),
            ("elapsed",      "経過"),
            ("eta_remain",   "残り時間"),
            ("eta_clock",    "完了予測"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(self._time_frame, text=label, width=14, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, padx=(6, 2), pady=2)
            val_lbl = ttk.Label(self._time_frame, text="—", anchor=tk.W)
            val_lbl.grid(row=i, column=1, sticky=tk.W, padx=(0, 6), pady=2)
            self._time_labels[key] = val_lbl

    # ── ポーリング ───────────────────────────────────────────────

    def _start_polling(self) -> None:
        def _poll() -> None:
            self._poll_once()
            self._parent.after(self._POLL_MS, _poll)

        self._parent.after(self._POLL_MS, _poll)

    def _poll_once(self) -> None:
        s = self._s

        # ── loss/lr バッファ取得 ────────────────────────────────
        pts = getattr(s, "_graph_points", None)
        loss_snapshot: list[tuple[int, float, float]] = []
        if pts is not None:
            lock = getattr(s, "_graph_lock", None)
            if lock:
                with lock:
                    loss_snapshot = list(pts)
            else:
                loss_snapshot = list(pts)

        # ── grad_norm バッファ取得 ──────────────────────────────
        gn_pts = getattr(s, "_grad_norm_points", None)
        gn_snapshot: list[tuple[int, float]] = []
        if gn_pts is not None:
            gn_lock = getattr(s, "_grad_norm_lock", None)
            if gn_lock:
                with gn_lock:
                    gn_snapshot = list(gn_pts)
            else:
                gn_snapshot = list(gn_pts)

        if loss_snapshot or gn_snapshot:
            steps  = [p[0] for p in loss_snapshot]
            losses = [p[1] for p in loss_snapshot]
            lrs    = [p[2] for p in loss_snapshot]
            gn_steps = [p[0] for p in gn_snapshot]
            gns      = [p[1] for p in gn_snapshot]
            self._redraw(steps, losses, lrs, gn_steps, gns)
            self._update_param_panel(s, steps, losses, lrs, gns)
            self._run_diagnostics(losses, gns)

        self._update_time_panel(s)

    # ── 再描画 ───────────────────────────────────────────────────

    def _redraw(
        self,
        steps: list[int],
        losses: list[float],
        lrs: list[float],
        gn_steps: list[int],
        gns: list[float],
    ) -> None:
        self._ax_loss.cla()
        self._ax_lr.cla()
        self._ax_gn.cla()
        self._configure_axes()
        if steps:
            self._ax_loss.plot(steps, losses, color="#3B82F6", linewidth=1.2)
            self._ax_lr.plot(steps, lrs, color="#10B981", linewidth=1.2)
        if gn_steps:
            self._ax_gn.plot(gn_steps, gns, color="#F59E0B", linewidth=1.2)
        self._canvas.draw_idle()

    def _configure_axes(self) -> None:
        self._ax_loss.set_title("Train Loss")
        self._ax_loss.set_xlabel("step")
        self._ax_loss.set_ylabel("loss")
        self._ax_lr.set_title("Learning Rate")
        self._ax_lr.set_xlabel("step")
        self._ax_lr.set_ylabel("lr")
        self._ax_gn.set_title("Grad Norm")
        self._ax_gn.set_xlabel("step")
        self._ax_gn.set_ylabel("grad_norm")

    # ── パネル更新 ───────────────────────────────────────────────

    def _update_param_panel(
        self,
        s: "TrainState",
        steps: list[int],
        losses: list[float],
        lrs: list[float],
        gns: list[float],
    ) -> None:
        if steps:
            self._param_labels["step"].configure(text=str(steps[-1]))
            self._param_labels["train_loss"].configure(text=f"{losses[-1]:.5g}")
            self._param_labels["lr"].configure(text=f"{lrs[-1]:.3e}")
        if gns:
            self._param_labels["grad_norm"].configure(text=f"{gns[-1]:.4g}")

        get_epoch = getattr(s, "get_current_epoch", None)
        if callable(get_epoch):
            epoch = get_epoch()
            if epoch > 0:
                self._param_labels["epoch"].configure(text=str(epoch))

    def _update_time_panel(self, s: "TrainState") -> None:
        start = getattr(s, "training_start_time", None)
        if start is None:
            return

        now     = datetime.datetime.now()
        elapsed = now - start
        self._time_labels["start"].configure(text=start.strftime("%H:%M:%S"))
        self._time_labels["elapsed"].configure(text=_format_duration(elapsed))

        # ETA は epoch 進捗ベースで概算する（musubi-tuner は epoch 単位で
        # save_every_n_epochs 等を扱うため、step 単位の精緻な推定は行わない）。
        get_epoch = getattr(s, "get_current_epoch", None)
        max_epochs_var = getattr(s, "max_train_epochs", None)
        if not callable(get_epoch) or max_epochs_var is None:
            self._time_labels["eta_remain"].configure(text="計算中...")
            self._time_labels["eta_clock"].configure(text="計算中...")
            return

        current_epoch = get_epoch()
        try:
            max_epochs = int(max_epochs_var.get())
        except (tk.TclError, ValueError):
            max_epochs = 0

        if current_epoch <= 0 or max_epochs <= 0:
            self._time_labels["eta_remain"].configure(text="計算中...")
            self._time_labels["eta_clock"].configure(text="計算中...")
            return

        per_epoch_sec = elapsed.total_seconds() / current_epoch
        remain_epochs = max(max_epochs - current_epoch, 0)
        remain_sec    = per_epoch_sec * remain_epochs
        remain_delta  = datetime.timedelta(seconds=remain_sec)
        eta_clock     = now + remain_delta

        self._time_labels["eta_remain"].configure(text=_format_duration(remain_delta))
        self._time_labels["eta_clock"].configure(text=eta_clock.strftime("%H:%M:%S"))

    # ── 自動診断 ─────────────────────────────────────────────────

    def _run_diagnostics(self, losses: list[float], gns: list[float]) -> None:
        messages: list[str] = []

        # grad_norm 急増判定
        if len(gns) >= self._GRAD_RECENT_WINDOW + 1:
            recent  = gns[-(self._GRAD_RECENT_WINDOW + 1):-1]
            latest  = gns[-1]
            recent_avg = sum(recent) / len(recent) if recent else 0.0
            if recent_avg > 0 and latest > recent_avg * self._GRAD_SPIKE_RATIO:
                if not self._diag_grad_spike_active:
                    messages.append(
                        f"[診断] grad_norm が急増しています "
                        f"({latest:.3f} / 直近平均 {recent_avg:.3f})。"
                        "LR低下またはmax_grad_norm調整を検討してください。"
                    )
                self._diag_grad_spike_active = True
            else:
                self._diag_grad_spike_active = False

        # Loss NaN/Inf 判定
        if losses:
            latest_loss = losses[-1]
            is_bad = latest_loss != latest_loss or latest_loss in (
                float("inf"), float("-inf")
            )
            if is_bad:
                if not self._diag_loss_nan_active:
                    messages.append(
                        "[診断] Train Loss が NaN/Inf になっています。"
                        "学習を停止することを推奨します。"
                    )
                self._diag_loss_nan_active = True
            else:
                self._diag_loss_nan_active = False

        for msg in messages:
            self._append_diag(msg)

    def _append_diag(self, msg: str) -> None:
        self._diag_text.configure(state=tk.NORMAL)
        self._diag_text.insert(tk.END, msg + "\n")
        self._diag_text.see(tk.END)
        self._diag_text.configure(state=tk.DISABLED)

    # ── 操作 ─────────────────────────────────────────────────────

    def _reset(self) -> None:
        s = self._s
        reset_fn = getattr(s, "reset_monitor_buffers", None)
        if callable(reset_fn):
            reset_fn()
        else:
            # 後方互換: reset_monitor_buffers 未実装の state.py への対応
            pts  = getattr(s, "_graph_points", None)
            lock = getattr(s, "_graph_lock", None)
            if pts is not None:
                if lock:
                    with lock:
                        pts.clear()
                else:
                    pts.clear()

        for lbl in self._param_labels.values():
            lbl.configure(text="—")
        for lbl in self._time_labels.values():
            lbl.configure(text="—")
        self._diag_grad_spike_active = False
        self._diag_loss_nan_active   = False
        self._diag_text.configure(state=tk.NORMAL)
        self._diag_text.delete("1.0", tk.END)
        self._diag_text.configure(state=tk.DISABLED)
        self._redraw([], [], [], [], [])

    def _stop_training(self) -> None:
        s = self._s
        if not s.is_running():
            s.log_fn("[WARN] 停止対象のプロセスがありません。")
            return
        from . import runner
        runner.stop_training(s)


def _format_duration(td: datetime.timedelta) -> str:
    """timedelta を 'H時間 M分 S秒' または 'M分 S秒' 形式の文字列に整形する。"""
    total = int(td.total_seconds())
    if total < 0:
        total = 0
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}時間 {m}分 {sec}秒"
    return f"{m}分 {sec}秒"
