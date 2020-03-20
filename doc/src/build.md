\page Build Build

現在allgebra Docker image上での利用を前提としています．
ここはmonolish本体の開発者向けのページです．

## CPU on Linux

* `make in-cpu` でallgebraをpullしてログインできます．
* /monolish/にmonolishがマウントされます．
* ビルドすれば使えるはずです．f8, o1で確認済．

Docker上で以下の手順でコンパイルしてインストールします．

```
> make cpu
> make install
```

共有ライブラリとして `libmonolish_cpu.so` が生成され， `$MONOLISH_DIR/lib/` にインストールされます．

## GPU on Linux
* `make in` でallgebraをpullしてログインできます．
CPUとGPUのsoファイルは別にしてあります．CPUとほとんど同じ手順でビルドします．

```
> make gpu
> make install
```

共有ライブラリとして `libmonolish_gpu.so` が生成され， `$MONOLISH_DIR/lib/` にインストールされます．

ヘッダはCPU, GPUで共通で， `$MONOLISH_DIR/include/` にインストールされます．

# 環境変数等
	* MONOLISH_DIR monolish 
	(インストール先，デフォルトは$(HOME)/lib/monolish)
	* BLAS_INC
   	(CBLASの場所， -I /usr/include/openblas/)
	* BLAS_LIB
	(CBLASの場所，-L/usr/lib64/ -lopenblas)

	* CUDA_INC
	(CUDAライブラリの場所，-I/usr/local/cuda-10.0/targets/x86_64-linux/include/)
	* CUDA_LIB
	(CUDAライブラリの場所，-L/usr/local/cuda-10.0/targets/x86_64-linux/lib/)

# Dependencies
CPU, GPU共通
		* make 
	   	* g++ 7.1以上 (C++17)
	   	* gfortran (Fortranとのリンクテストのみ)
		* python3 (optional, log集計用)
		* python3-yaml (optional, log集計用)
		* python3-numpy (optional, log集計用)

GPU
		* nvptx-tools ()
		* offload-nvptx対応g++
		* cuda-cublas-dev-10-0
		* cuda-cudart-dev-10-0
		* cuda-compiler-10.0
		* cuda-cusolver-dev-10-0
		* cuda-cusparse-dev-10-0 
		* cuda-nvprof-10-1 (optional)

CPU
		* libopenblas-dev

## 制約等
* c++17が必要です
* cuda-10.0以外の環境ではテストしていません
* gcc + OpenACCが必要です．
