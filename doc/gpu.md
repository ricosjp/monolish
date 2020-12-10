# GPU programming {#gpu_md}

# はじめに
monolishの各クラス(vector, matrix)は`send()`関数を用いることでGPUにマッピングされる．
GPUにマッピングされたデータは`recv()`または`device_free`することによってGPUから解放される．

`libmonolish_cpu.so`をリンクした場合， `send()` や `recv()`は何も行わないため，コードの共通化は可能である．

`libmonolish_gpu.so`をリンクした場合， `send()` によってGPUにデータを送ることが出来るようになる．
GPUに転送済のデータかどうかは `get_device_mem_stat()` 関数によって得られる．

* CPUでデータを生成し，
* GPUで計算し，
* 最後にデータをCPUに受け取る

という流れを意識することが転送を減らす上で重要である．

GPUにデータが転送された変数は挙動が異なり，ベクトルや行列の要素に対する演算はできなくなる．

また，演算時にCPU/GPUどちらのデータに処理が反映されるかも注意が必要である．

本ページでは`get_device_mem_stat()`がtrueのときの各関数の挙動について説明する

# 各クラスのGPUにおける挙動の違い
## monolish::vector

| Operation                                | where to run | Description                    | memo                          |
|------------------------------------------|--------------|--------------------------------|-------------------------------|
| BLAS演算                                 | GPU          | axpy(), matvec()など               |                               |
| 要素参照                                 | Error        | operator[], at(), insert()など |                               |
| 要素演算                                 | GPU          | elemadd()など                    |                               |
| 代入演算子                               | GPU          | operator=, operator+=など      | サイズが異なると死ぬ          |
| copy関数                                 | CPU/GPU      | y = x.copy()                   | 転送が発生する可能性がある    |
| monolish::vectorでのコピーコンストラクタ | CPU/GPU      | コピーコンストラクタ           | PU/GPUの両方の状態がコピー    |
| 比較演算子                               | GPU          | operator==, operator!=         |                               |
| 情報取得                                 | CPU          | size()など                     |                               |
| resize()                                 | Error        | ベクトルサイズの変更           |                               |
| print\_all()                             | CPU          | ベクトルの全出力               | デバッグ用にCPUのデータを吐く |

## monolish::matrix::COO

計算用の関数を持たないためGPUでは扱えない． `send` や `recv` 関数も使えない

## monolish::matrix::CRS

| Operation                                | where to run | Description                    | memo                          |
|------------------------------------------|--------------|--------------------------------|-------------------------------|
| BLAS演算                                      | GPU     | SpMVなど                       |                               |
| 要素参照                                      | Error   | operator[], at(), insert()など |                               |
| 要素演算                                      | GPU   | elemadd()など                  |                               |
| 行ベクトル・列ベクトルの取得                  | GPU     | get\_diag()など                |                               |
| 算術演算子                                    | GPU     | operator+など                  |                               |
| 代入演算子                                    | GPU     | operator=, operator+=など      | サイズが異なると死ぬ          |
| copy関数                                      | CPU/GPU | y = x.copy()                   | 転送が発生する可能性がある    |
| monolish::matrix::CRSでのコピーコンストラクタ | CPU/GPU | コピーコンストラクタ           | PU/GPUの両方の状態がコピー    |
| 情報取得                                      | CPU     | size(),get\_rowなど            |                               |
| print\_all()                                  | CPU     | ベクトルの全出力               | デバッグ用にCPUのデータを吐く |

## monolish::matrix::Dense

| Operation                                | where to run | Description                    | memo                          |
|------------------------------------------|--------------|--------------------------------|-------------------------------|
| BLAS演算                                      | GPU     | SpMVなど                       |                               |
| 要素参照                                      | Error   | operator[], at(), insert()など |                               |
| 要素演算                                      | GPU   | elemadd()など                  |                               |
| 行ベクトル・列ベクトルの取得                  | GPU     | get\_diag()など                |                               |
| 算術演算子                                    | GPU     | operator+など                  |                               |
| 代入演算子                                    | GPU     | operator=, operator+=など      | サイズが異なると死ぬ          |
| copy関数                                      | CPU/GPU | y = x.copy()                   | 転送が発生する可能性がある    |
| monolish::matrix::CRSでのコピーコンストラクタ | CPU/GPU | コピーコンストラクタ           | PU/GPUの両方の状態がコピー    |
| 情報取得                                      | CPU     | size(),get\_rowなど            |                               |
| print\_all()                                  | CPU     | ベクトルの全出力               | デバッグ用にCPUのデータを吐く |
