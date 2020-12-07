<!--
Please Keep this comment on the top of this file

How to write Changelog
-----------------------

https://keepachangelog.com/ja/1.0.0/ に基づいて記述していく

- Merge Request毎に記述を追加していく
- 何を変更したかを要約して書く。以下の分類を使う
  - Added      新機能について。
  - Changed    既存機能の変更について。
  - Deprecated 間もなく削除される機能について。
  - Removed    今回で削除された機能について。
  - Fixed      バグ修正について。
  - Security   脆弱性に関する場合。
- 日本語でも英語でも良い事にする

-->

Unreleased
-----------
### Added
- matrix copyのテストを追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140
- src/internalに配列に対する基本演算のコードを実装 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139
- make testでmake test-cpuとmake-gpuを両方実行するようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139

### Changed 
- vector四則演算の裏側をinternalに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140
- ドキュメントの呼び出し関数と呼び出しライブラリ一覧を修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139
- src/monolish_internal.hppをsrc/internal/monolish_internal.hppに移動 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139

0.7.0 - 2020/12/04
-----------
### Added
- cmakeでclang11+GPUに対応した https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/137

### Changed 
- CIのartifactの寿命を360分に延長 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138
- benchmarkの高速化のために乱数値のベクトルでなく定数ベクトルに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138
- clang11gcc7コンテナに合わせてベンチマークサイズを変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138
- monolishコンテナをclangでビルドするようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/136
- clangに合わせてtest/lang/fortranのオプションに-fPIEをつけるようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/136

### Removed
- cmake作成前に一時的に作成したMakefile.clang-gpuを削除 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138

0.6.2 - 2020/11/17
-----------
### Added
- benchmark結果のURLをDoxygenに記載 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/134
- benchmarkで演算の種類(kind)を出力するようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/129

### Changed
- benchmarkの出力ディレクトリ名をハッシュ名だけに戻したhttps://gitlab.ritc.jp/ricos/monolish/-/merge_requests/128

### Fixed
- vectorのbenchmarkサイズと繰り返し回数をメモリエラーが起きない範囲に調整 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/131
- vectorのbenchmarkがfailしてもCIでerror扱いにならないのを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/131

0.6.1 - 2020/11/15
-----------
### Changed 
- benchmarkの出力ディレクトリ名変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/122
- benchmarkの測定サイズ変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/121
- benchmarkでサイズの繰り返しをヘッダでまとめで定義するようにしたhttps://gitlab.ritc.jp/ricos/monolish/-/merge_requests/124

### Added
- benchmarkでpipeline_idを出力するようにする https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/119

### Fixed
- CG法のベクトルの更新がおかしいのを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/127
- ベンチマークの測定スクリプトのバグ修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/121
- cusparseの関数が非同期で実行されているようなのでsyncを追加した https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/120
- benchmarkをtagsとschedulesでは実行しないようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/116
- benchmarkの出力ファイルの末尾に半角スペースが入っているバグを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/117
- benchmarkの出力ファイルの末尾にtabが入っているバグを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/115
- matadd, mscalのbenchmarkの出力ファイルのバグを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/114


0.6.0 - 2020/11/04
-----------
### Added
- monolish_log_viewer にtest追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/118
- ベンチマークのディレクトリを作成，masterでのみ実行するようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/112
- benchmark/にtsvを置くとCIの最後でベンチマーク結果としてアップロードされるようになった https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/111
- version 0.0.4の `test/` を `benchmark/` として復活させたhttps://gitlab.ritc.jp/ricos/monolish/-/merge_requests/110
- CHANGELOG.md (このファイル) の追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/106
- GitLab CI で Merge Request 毎に origin/master から CHANGELOG.md に更新があるかチェックする https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/105

### Changed
- `pyproject.toml` と `__init__.py` を作成して monolish_log_viewer を Python パッケージにする https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/107
- タグ付けのときはkeep-changelogを走らないようにするhttps://gitlab.ritc.jp/ricos/monolish/-/merge_requests/113
- cmakeでNVPTXのオプションに-misa=sm_35と-lmを付けた https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/109
- test/logger/logging にある Python スクリプト群を Project TOP に移動させる https://gitlab.ritc.jp/ricos/monolish/-/issues/325
- loggerの3層以上のアルゴリズム改善 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/103
- Base allgebra image switched from 20.10.0 to 20.10.1 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/105
