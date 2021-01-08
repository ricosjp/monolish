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
- add fill function https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/193
- add util::build_with functions https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/192
- add xpay test https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/192
- add nrm1 and get_residual_l2(Dense) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/191
- add Frank matrix creation and eigenvalue calculation routine https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/189
- add jacobi solver https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/188
- add jacobi preconditioner https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/188
- add vml::reciprocal https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/188
- add LOBPCG eigensolver https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/88 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/194
- Support MKL SpMV and SpMM https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/185
- install monolish_log_viewer on monolish container https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/183
- add install-sxat install-a64fx target in makefile https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181
- define four benchmarks {intel-MKL, intel-OSS, AMD-OSS, GPU-MKL} https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/179
- add oss test https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178
- add makefile target `make oss-cpu` `make oss-cpu` `make mkl-cpu` `make mkl-gpu` https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178

### Changed 
- CRS.print_all() output matrixmarket format https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/196
- support print_all() on GPU https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/196
- move equation::solver and equation::precondition to solver::solver and solver::precondition https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/190
- exclude src/internal doxygen https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/182
- update allgebra 20.12.2 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/182
- include algorithm in internal.hpp https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181
- change name sx->sxat, fx->a64fx https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181
- deploy benchmark only:master -> only:schedules(weekly) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/179
- benchmark only:master -> only:schedules(weekly) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178
- change CI job prefix name [ops]-[arch] -> [arch]-[ops] https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178

### Fixed
- fix LOBPCG iteration logic https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/195
- fix sxat, a64fx makefile bugs https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181

0.9.1 - 2020/12/28
-----------
### Added
- add vecadd/vecsub doxygen comments https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/177
- CG, BiCGSTABでBREAKDOWNしたりresidualがNaNになったときの判定を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/175
- BiCGSTABの実装を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/174

### Changed 
- BiCGSTABを実装済としてDoxygenに反映, update doxygen project vesion to 0.9.1 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/177
- CG,BiCGSTABでA,x,bのGPU Statusが一致していなければerrorになるようにしたhttps://gitlab.ritc.jp/ricos/monolish/-/merge_requests/176
- update allgebra 20.10.1 -> 20.12.1 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/173
- CIのRunner指定をhostnameからGPUのsmタグに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/171

### Fixed
- test/equationのminiterを0に設定 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/174
- Doxygenのmarkdownのtableが崩れているのを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/172
- monolish_log_viewerの連続処理カウント処理を修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/169

0.9.0 - 2020/12/21
-----------
### Added
- VMLのDoxygenコメントとmarkdownへの反映 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/166
- monolish_log_viewer のライセンスを Apache-2.0 に設定 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/167
- CRSに数学関数を追加 (min, max以外) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/164
- Denseに数学関数を追加 (min, max以外) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/163
- vectorに数学関数を追加 (min, max以外) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/162
- internalに数学関数を追加 (min, max以外) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/160
- Doxygenのfunction listにvmlに実装予定の数学関数の一覧を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/159
- internalとvmlにvtanhのコードを実装 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/157

### Changed 
- internalのvdivをMKLに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/161
- CIのRunner指定をホスト名でなくMacアドレスのタグ名に変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/158
- test, benchmarkにvml::vtanhに変更．元の各クラスのメンバ関数としての実装は削除 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/157
- testにscalar-matrixのVMLがなかったので追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/157

### Fixed
- powerの乱数の範囲を1~2に変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/165

0.8.0 - 2020/12/17
-----------
### Added
- VMLのDoxygenコメントを追加，Doxygenバージョンを0.8.0へ https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/156
- matrixに一致判定関数を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/156
- matrix四則演算関数を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/149
- matadd/matsub関数を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/149
- vecadd/vecsubを追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/148
- vector四則演算関数を追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/147
- matrix subを追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/146

### Changed 
- test/とbenchmarkをcommon, vml, blasの3つに分割して整理 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/151
- 四則演算関数をmonolish::vml名前空間, src/vml/, include/monolish_vml.hppに移動 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/151
- matadd/をmataddsub/に変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/149
- test/benchmarkのvector_commonをoperatorでなく四則演算関数に変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/147
- matrix add/subでdoubleからfloatを作るようにファイル構成を変更  https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/146

### Removed
- すべてのクラスの四則演算のoperatorを削除, test,benchmarkも同様に削除 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/153

0.7.1 - 2020/12/10
-----------
### Added
- matrix copyのテストを追加 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140
- src/internalに配列に対する基本演算のコードを実装 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139
- make testでmake test-cpuとmake-gpuを両方実行するようにした https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139

### Changed 
- vector,CRS,denseの現状をDoxygenに反映 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/145
- cmake中の環境変数を MKL_ROOT から MKLROOT に変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/144
- CRSのadd, copy, scalの裏側をinternalに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/142
- denseのadd, copy, scalの裏側をinternalに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/141
- CIのコンテナ名を変数にして上から再設定できるようにした(+gitlabから変数をRICOSのコンテナレジストリにした) https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/141
- vectorのoperator==をCPU/GPU両方のデータの完全一致でtrueに変更 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140
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
- monolish_log_viewerにlintをかけた https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/135
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
