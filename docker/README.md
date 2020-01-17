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
だいたい10^10～10^20くらいの結構大きい値が答えになります．

GPUでゼロとか，10^-20とかの小さい値が返ってきた場合はおそらく失敗しています．
`Docker --gpus all` をつけていないとか，GPUが認識していないとかです．
`nvidia-smi`で確認して下さい．それでだめなら菱沼まで．


## sparse LU
GPUのみ．cusolverを用いたsparse LU．slu.cppを参照．
