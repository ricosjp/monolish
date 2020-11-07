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
### Fixed
- 出力ファイルの末尾にtabが入っているバグを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/115
- matadd, mscalのbenchmarkの出力ファイルのバグを修正 https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/114

0.6.0 - 2020/11/04
-----------
### Added
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
