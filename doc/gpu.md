# GPU programming {#gpu_md}

# はじめに
monolishの各クラス(vector, matrix)は`send()`関数を用いることでGPUにマッピングされる．
GPUにマッピングされたデータは`recv()`または`device_free`することによってGPUから解放される．

`libmonolish_cpu.so`をリンクした場合， `send()` や `recv()`は何も行わないため，コードの共通化は可能である．

`libmonolish_gpu.so`をリンクした場合，ほとんどの演算機能はGPUに転送しないと使えない．
GPUに転送済のデータかどうかは `get_device_mem_stat()` 関数によって得られる．

* CPUでデータを生成し，
* GPUで計算し，
* 最後にデータをCPUに受け取る

という流れを意識することが転送を減らす上で重要である．

本ページでは`get_device_mem_stat()`がtrue / falseの各関数の挙動について説明する

# 各クラスのCPU/GPUにおける挙動の違い
## monolish::vector


`get_device_mem_stat()` == true

| Operation                                | Arch.   | Description                    | memo                          |
|------------------------------------------|---------|--------------------------------|-------------------------------|
| BLAS演算                                 | GPU     | axpy, dotなど                  |                               |
| 要素参照                                 | Error   | operator[], at(), insert()など |                               |
| 算術演算子                               | GPU     | operator+など                  |                               |
| 代入演算子                               | GPU     | operator=, operator+=など      | サイズが異なると死ぬ          |
| copy関数                                 | CPU/GPU | y = x.copy()                   | 転送が発生する可能性がある    |
| 比較演算子                               | GPU     | operator==, operator!=         |                               |
| 情報取得                                 | CPU     | size()など                     |                               |
| monolish::vectorでのコピーコンストラクタ | CPU/GPU | コピーコンストラクタ           | PU/GPUの両方の状態がコピー    |
| resize()                                 | Error   | ベクトルサイズの変更           |                               |
| print\_all()                             | CPU     | ベクトルの全出力               | デバッグ用にCPUのデータを吐く |
| 一致判定                                 | Error   | operator==, operator!=         |                               |

`get_device_mem_stat()` == false

| Operation                                | Arch.   | Description                    | memo                          |
|------------------------------------------|---------|--------------------------------|-------------------------------|
| BLAS演算                                 | Error   | axpy, dotなど                  |                               |
| 要素参照                                 | CPU     | operator[], at(), insert()など |                               |
| 算術演算子                               | Error   | operator+など                  |                               |
| 代入演算子                               | CPU     | operator=, operator+=など      | サイズが異なると死ぬ          |
| copy関数                                 | CPU     | y = x.copy()                   |                               |
| 比較演算子                               | CPU     | operator==, operator!=         |                               |
| 情報取得                                 | CPU     | size()など                     |                               |
| monolish::vectorでのコピーコンストラクタ | CPU     | コピーコンストラクタ           | これはCPU/GPUだろう           |
| resize()                                 | CPU     | ベクトルサイズの変更           |                               |
| print\_all()                             | CPU     | ベクトルの全出力               |                               |
| 一致判定                                 | CPU     | operator==, operator!=         |                               |

## monolish::matrix::COO

計算用の関数を持たないためGPUでは扱えない． `send` や `recv` 関数も使えない

## monolish::matrix::CRS
`get_device_mem_stat()` == true

| Operation                                     | Arch.   | Description                    | memo                          |
|-----------------------------------------------|---------|--------------------------------|-------------------------------|
| BLAS演算                                      | GPU     | SpMVなど                       |                               |
| 要素参照                                      | Error   | operator[], at(), insert()など |                               |
| 行ベクトル・列ベクトルの取得                  | GPU     | get\_diag()など                |                               |
| 算術演算子                                    | GPU     | operator+など                  |                               |
| 代入演算子                                    | GPU     | operator=, operator+=など      | サイズが異なると死ぬ          |
| copy関数                                      | CPU/GPU | y = x.copy()                   | 転送が発生する可能性がある    |
| 情報取得                                      | CPU     | size(),get\_rowなど            |                               |
| monolish::matrix::CRSでのコピーコンストラクタ | CPU/GPU | コピーコンストラクタ           | PU/GPUの両方の状態がコピー    |
| print\_all()                                  | CPU     | ベクトルの全出力               | デバッグ用にCPUのデータを吐く |

`get_device_mem_stat()` == false

| Operation                                     | Arch.   | Description                    | memo                          |
|-----------------------------------------------|---------|--------------------------------|-------------------------------|
| BLAS演算                                      | Error   | SpMVなど                       |                               |
| 要素参照                                      | Error   | operator[], at(), insert()など |                               |
| 行ベクトル・列ベクトルの取得                  | CPU     | get\_diag()など                |                               |
| 算術演算子                                    | Error   | operator+など                  |                               |
| 代入演算子                                    | CPU     | operator=, operator+=など      |                               |
| copy関数                                      | CPU     | y = x.copy()                   |                               |
| 情報取得                                      | CPU     | size(),get\_rowなど            |                               |
| monolish::matrix::CRSでのコピーコンストラクタ | CPU     | コピーコンストラクタ           |                               |
| print\_all()                                  | CPU     | ベクトルの全出力               |                               |
