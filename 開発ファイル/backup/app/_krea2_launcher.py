# _krea2_launcher.py — musubi-tuner Krea2 学習ランチャー
# GUIアプリが自動生成します。手動編集不要。
# Windows multiprocessing spawn 対応: exec() を使用しない。
import sys
import os
import runpy
from multiprocessing import freeze_support

_SRC  = 'E:/Omusubi_krea2_gui/musubi-tuner/src'
_ROOT = 'E:/Omusubi_krea2_gui/musubi-tuner'

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_ROOT)

if __name__ == '__main__':
    freeze_support()
    runpy.run_module(
        'musubi_tuner.krea2_train_network',
        run_name='__main__',
        alter_sys=True,
    )
