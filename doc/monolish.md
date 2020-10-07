# monolish {#mainpage}
monolish is MONOlithic LIner equation Solvers for Highly-parallel architecture.

# Build and Install
- 現状ではCUDA関係のライブラリのバージョンなどの制約から **Docker環境以外での利用は考えてません** ．
- monolishが配布しているコンテナを利用するか，DockerFileから `FROM` して利用します．
- コンテナのリポジトリはこちら:[gitlab](https://gitlab.ritc.jp/ricos/monolish/container_registry)
- サンプルコードが `/opt/ricos/monolish/$VERSION/samples` に入っている．
- ライブラリのリンク設定によってCPUとGPUを切り替える．
- 環境変数などはすべて通してあるため，自分で作成したプログラムは `-lmonolish_cpu` か `-lmonolish_gpu` をつければ動くはず．
- ただし現在CPU版はロクなソルバがないので使う意味があまり無いです
- なお，monolishは `/opt/ricos/monolish/$VERSION/` にインストールされている．

自分でビルドする場合(非推奨)
- [ここを見る](@ref build_md) 


# How to programming with monolish
- [概要](@ref qstart_md) 
- [GPU Programming](@ref gpu_md) 

# 各関数の実装
Intel, NVIDIA, OSSの説明をここに書く

それぞれで何が呼ばれているかは [ここ](@ref oplist_md) をみる．

# 各格納形式の機能
[ここ](@ref MatUtil_list_md) をみる．


# Logger
logファイルの例もCIで生成されるようになりました

`test/logger/logging` の結果が出るようになっています．

- [CPU](https://ricos.pages.ritc.jp/monolish/logging_result_cpu.html)
- [GPU](https://ricos.pages.ritc.jp/monolish/logging_result_gpu.html)


### そのほか
開発の思想などは以下のMTG資料を見てください

第1回MTG: [GSlides Link](https://docs.google.com/presentation/d/1LzTvWe_b_oKFHR2HP7gd1ds7nLxLUi2ncWVo9qk0x0c/edit?usp=sharing)

第2回MTG: [GSlides Link](https://docs.google.com/presentation/d/1bgzDkHm5AHRyxxj2mM09zGMT9P9IkH21UNLrKanhyG0/edit?usp=sharing)

ver1.0 MTG: [GSlides Link](https://docs.google.com/presentation/d/12LJXbFmAmKcEWtkIBCZm_klpqmAP6MIuvYCRAZnvwqQ/edit?usp=sharing)
