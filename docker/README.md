# monolish docker
Allgebraをベースにmonolishの共有ライブラリやヘッダファイルがインストールされたイメージ

monolishが/opt/ricos/monolish/$VERSION/にインストールされています．

# サンプルコード
サンプルコードも/opt/ricos/monolish/$VERSION/samplesに入っています．

makeして動くかを確認して下さい．

## 内積
CPU/GPUでOpenBLAS/cuBLASを用いた内積をする．dot.cppを参照．

答え合わせは入れてません，
出力は実行時間とN=10^4の倍精度乱数ベクトルの内積です．



## sparse LU
GPUのみ．cusolverを用いたsparse LU．slu.cppを参照．
