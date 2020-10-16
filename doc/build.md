# Build monolish {#build_md}

現在allgebra Docker image上での利用を前提としています．
ここはmonolish本体の開発者向けのページです．

ビルド可能なコンテナとしてallgebraがあります．コンテナにログインするためのコマンドはMakefileに書いてあるため，
以下のコマンドで使用できます．

* CPUの場合は `make in-cpu` 
* GPUの場合は `make in` 

コンテナには/monolish/にmonolishがマウントされます．

monolish自体のビルドにはGNU MakeとCmakeが使えます．

CPUでは共有ライブラリとして `libmonolish_cpu.so` 
GPUでは `libmonolish_gpu.so` が生成されます．

ヘッダはCPU, GPUで共通です．

## make

インストール場所は `$MONOLISH_DIR/lib/` です．

### CPU on Linux
Docker上で以下の手順でコンパイルしてインストールします．

```
> make cpu
> make install
```


### GPU on Linux
Docker上で以下の手順でコンパイルしてインストールします．

```
> make gpu
> make install
```

## cmake
build/など作ってcmakeでコンパイルしてください．
オプションは以下の2つが使えます．

- -DBUILD\_GPU={ON/OFF}
- -DCMAKE\_INSTALL\_PREFIX=[dir]

## 環境変数等
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

## Dependencies
CPU, GPU共通
		* make 
	   	* g++ 5.1以上 (C++14)
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
* c++14が必要です
* cuda-10.0以外の環境ではテストしていません
* gcc + OpenACCが必要です．
