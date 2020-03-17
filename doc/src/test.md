\page howto How to use

# ユーザ向け：git cloneからサンプルの実行まで

### 実行環境
現状ではCUDA関係のライブラリのバージョンなどの成約から **Docker環境以外での利用は考えてません** ．
monolishが配布しているコンテナを利用するか，DockerFileから `FROM` して利用します．

コンテナのリポジトリはこちら:[gitlab](https://gitlab.ritc.jp/ricos/monolish/container_registry)

サンプルコードが `/opt/ricos/monolish/$VERSION/samples` に入っている．
ライブラリのリンク設定によってCPUとGPUを切り替える．
環境変数などはすべて通してあるため，自分で作成したプログラムは `-lmonolish_cpu` か `-lmonolish_gpu` をつければ動くはず．
ただし現在CPu版はロクなソルバがないので使う意味があまり無いです

なお，monolishは `/opt/ricos/monolish/$VERSION/` にインストールされている．


# How to programming with monolish
重要な機能の使い方について簡単に紹介する

## 基本のクラス

### monolish::vector クラス
想定利用方法: ほとんどstd::vectorと同じです．演算子などもCPU / GPUで使えます．
([関連コード][vec])

### monolish::matrix::COO クラスおよび monolish::matrix::CRS クラス
Sorted COOをMM形式のファイル，または{double val, int row, int col}の3本の配列から作成し，CRSのコンストラクタにCOOを入れることで変換．
([関連コード][mat])

## ソルバ関係

### monolish::equation::LU クラス
CRS形式の疎行列とベクトルを入力すればcuSolverのsparse LUを実行できる(CPUのみ)
([関連コード][slu])

### monolish::equation::cg クラス
CRS形式の疎行列とベクトルを入力すればCG法を実行できる気がする．．．
([関連コード][cg])

## ログ関係の関数

### monolish::set_log_level() 関数
備考: 0が出力なし，1がソルバ全体時間のみ，2がBLASも出力，3がUtil関係含めて全部出力

### monolish::set_log_filename() 関数
備考: 設定しなければstandard I/Oに出力する

## かなしいこと
* iterationのログ関係がないです
* 終了コード周りがちょっと雑です
* メモリ管理がアレです．たぶんリークしてるので長時間回すと死にます (これは直す+CIにvalgrindを入れます)
* MPIはまだまだ．．
* COOの操作巻数が少なくて複雑な前処理がきついです．．．
* operator系はGPU対応していますが正直転送するので遅いです．．．

[vec]: https://ricos.pages.ritc.jp/monolish/d8/df5/vector__common_8cpp_source.html
[mat]: https://ricos.pages.ritc.jp/monolish/d8/df5/matrix__common_8cpp_source.html
[slu]: https://ricos.pages.ritc.jp/monolish/d9/d44/slu_8cpp_source.html
[cg]: https://ricos.pages.ritc.jp/monolish/d5/d1e/test_2equation_2cg_2cg_8cpp_source.html
