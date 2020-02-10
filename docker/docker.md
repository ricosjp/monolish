# monolish docker
Allgebraをベースにmonolishの共有ライブラリやヘッダファイルがインストールされたイメージ

monolishが/opt/ricos/monolish/$VERSION/にインストールされています．

# サンプルコード
サンプルコードも/opt/ricos/monolish/$VERSION/samplesに入っています．

`make` または`make gpu` して動くかを確認して下さい．

サンプルは本家の`test/` を最小化したもので，
答え合わせやサイズを入力する引数は外してあります．
何もつけずにそのまま実行すれば動くはずです．

## 内積 (dot.cpp)
CPU/GPUでOpenBLAS/cuBLASを用いた内積をする．

出力は実行時間とN=10^4の倍精度乱数ベクトルの内積です．\
だいたい10^10～10^20くらいの結構大きい値が答えになります．

GPUでゼロとか，10^-20とかの小さい値が返ってきた場合はGPUのメモリ確保や転送が失敗しています．\
`Docker --gpus all` をつけていないとか，GPUが認識していないとかです．
`nvidia-smi`で確認して下さい．それでだめなら菱沼まで．


## sparse LU (slu.cpp)
GPUのみ．cusolverを用いたsparse LU．

test.mtxを読み込んで連立一次方程式を解きます．

出力は実行時間と答えのベクトルです．\
答えが1になるように作っているので，1以外になったら失敗です．
