import sys, os
sys.path.insert(0, r'E:\Omusubi_krea2_gui\musubi-tuner\src')
os.chdir(r'E:\Omusubi_krea2_gui\musubi-tuner')
with open(r'E:\Omusubi_krea2_gui\musubi-tuner\src\musubi_tuner\krea2_train_network.py', encoding='utf-8') as _f:
    _code = compile(_f.read(), _f.name, 'exec')
exec(_code, {'__name__': '__main__', '__file__': r'E:\Omusubi_krea2_gui\musubi-tuner\src\musubi_tuner\krea2_train_network.py'})
