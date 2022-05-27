# Release flow {#release_flow}

1. Changelogを直すコミットを手動で行う（Unreleaseを次のバージョンにする）
    - このコミットに手動でタグを打つ。そうすると自動的にこれのCI（trigger = tag）でリリース用debファイルをGitLab CIのartifactとして作成
    - ↑のdebを使ってコンテナを作成+GitHub Container Registryにpush
1. （CIが終わるのを待つ）このタグがGitHubに自動でミラーされるので手動でGitHub Releaseを作成
1. 手動でGitLab CI gather_deb ジョブのartifactからdebを手動でダウンロードしてGitHub Releaseに添付する

リリース後、次バージョンの開発開始処理．マージリクエスト start_0.15.2 などでやる

1. READMEのversion tableに今リリースした分を手動で追加する
    - e.g., 0.14.2の列をベースに0.15.0の列を作る
1. CMakeLists.txtのバージョンを次のリリース予定の手動でバージョンにあげる
    - e.g., 0.15.0をリリースしたら0.15.1か0.16.0にしておく
