# monolish docker
Allgebraをベースにmonolishの共有ライブラリやヘッダファイルがインストールされたイメージ

monolishが/opt/ricos/monolish/$VERSION/にインストールされています．

# サンプルコード
サンプルコードも/opt/ricos/monolish/$VERSION/samplesに入っています．

makeして動くかを確認して下さい．

## 内積
CPU/GPUでOpenBLAS/cuBLASを用いた内積をする．dot.cppを参照．

出力は実行時間とN=10^4の倍精度乱数ベクトルの内積です．
サンプルに答え合わせは入れてませんが，
だいたい10^10～10^20くらいの結構大きい値が答えになります．

GPUでゼロとか，10^-20とかの小さい値が返ってきた場合はGPUのメモリ確保や転送が失敗しています．
`Docker --gpus all` をつけていないとか，GPUが認識していないとかです．
`nvidia-smi`で確認して下さい．それでだめなら菱沼まで．


## sparse LU
GPUのみ．cusolverを用いたsparse LU．slu.cppを参照．

test.mtxを読み込んで連立一次方程式を解きます．
出力は実行時間と答えのベクトルです．
答えが1になるように作っているので，1以外になったら失敗です．
