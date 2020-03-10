\page Build Build

本ライブラリは現在allgebra Docker image上での利用を前提としています．
ここはmonolish本体の開発者向けのページです．

## CPU on Linux

* `make in` でallgebraをpullしてログインできます．
* /monolish/にmonolishがマウントされます．
* ビルドすれば使えるはずです．o1で確認済．


Docker上で以下の手順でコンパイルしてインストールします．

```
> make cpu
> make install
```

共有ライブラリとして `libmonolish_cpu.so` が生成され， `$MONOLISH_DIR/lib/` にインストールされます．

## GPU on Linux
CPUとGPUのsoファイルは別にしてあります．CPUとほとんど同じ手順でビルドします．

```
> make gpu
> make install
```

共有ライブラリとして `libmonolish_gpu.so` が生成され， `$MONOLISH_DIR/lib/` にインストールされます．

ヘッダはCPU, GPUで共通で， `$MONOLISH_DIR/include/` にインストールされます．

## 成約等
* c++17が必要です
* cuda-10.0以外の環境ではテストしていません
* gcc + OpenACCが必要です．
