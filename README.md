# monolish: MONOlithic Liner equation Solvers for Highly-parallel architecture
monolish is a linear solver library that monolithically fuses variable data type, matrix structures, matrix data format, vender specific data transfer APIs, and vender specific numerical algebra libraries.
monolish is a vendor-independent open source library written in C ++ that aims to be grand unified linear algebra library on any hardware.

# Feature (Policyか？)
monolish let developper be oblivious about:
- Argument data type of matrix/vector operations
- Matrix structure / storage format
- [Various processor which execute library  (Intel/NVIDIA/AMD/ARM etc.) ][oplist]
- [Vender specific data transfer API (host RAM to Device RAM)][gpu]
- Cumbersome package dependency
- Perormance tuning and visualization

各特徴を説明するファイルを作ってリンクを貼る

[oplist]: doc/operation_list.md
[gpu]: doc/gpu.md


# Build and Install
## Download binary
リンクを貼る

## Docker
[allgebra](https://github.com/ricosjp/allgebra)

Install path is `$MONOLISH_DIR` 

### CPU
```
make cpu
make install
```

### GPU
```
make gpu
make install
```

# Support
Supportの受け方をここに書く(issueを書いてね)
If you have any question, bug to report or would like to propose a new feature, feel free to create an [issue][issue] on GitHub.

[issue]: http://gogo-gomachan.com/charactor/

# Contributing
Contributionに必要な情報をここに書く

```
monolishはchangelogに変更がない場合とclang-formatかけてない場合にCIでwarningがでる．

- clang formatはTOPDIRで make format するとgit addされてるファイル全部に自動でかかります
- changelogは `CHANGELOG.md` に主要な変更とマージリクエストの番号を書く
```

# Licensing
monolish is available under the Apache License

ライセンスファイルへのリンクをここに貼る



# memo
- Oblivious about platform
- Oblivious about High performance(MPI, Multi-threading, SIMD)
- Don’t reinvent the wheel (BLAS/SparseBLAS, Matrix Asm., Math func.,)
- Performance Logging to Find the bottleneck (logger)
- Build and Package Managing (cmake, Docker)
- zero-overhead
Don't implicitly allocate memory??
- user friendly communication I/F
- Don't require many code changes due to hardware changes (Intel, AMD, Power, ARM, A64fx, SXAT, NVIDIA GPU, AMD GPU...)
- 簡単にかけるCommunication I/F (GPU/MPI, send-recv)


======
- Don't implement functions whose performance is unknown
- Continuous Integration／Continuous Delivery/Continuous Benchmark (monolish performance viewer)

- easy to implement other Language FFI (C++ and other language I/F)


- Don't suffer from dependency resolution

どこでも動く：
- BLAS: 150+ functions (163?)
- LAPACK: 1000+ functions (1302?)

Intelだけ：
- MKL2020: 10000+ functions (14850?)

NVIDIAだけ：
- CUBLAS/CUSPARSE/CUSOLVER/cuFFT cuda11.2: 1000+ functions (1112?)


- C++のパッケージマネージメントをするためにDockerを使う
- 性能を保証するためにCIでベンチマークして可視化する
- BLAS/LAPACKとMKLの差はmonolishが埋める
- 型や行列形状に依存した関数名をやめてtemplate化する
- ユーザが想像できないメモリを確保しない


# MTG資料

初期設計資料 [GSlides](https://docs.google.com/presentation/d/16JvP7bTtxmfMP9hqflB7FVDrxueYxYa5U2PT-SkqB20/edit?usp=sharing)

Matrix Format [Gslides](https://docs.google.com/presentation/d/1wqyw9CmlHar84WxTgnoULn0_ZHZ7IxkUnLa_HkIwVQo/edit?usp=sharing)

ver1.0 MTG: [GSlides Link](https://docs.google.com/presentation/d/12LJXbFmAmKcEWtkIBCZm_klpqmAP6MIuvYCRAZnvwqQ/edit?usp=sharing)

# 関連URL

https://ricos.pages.ritc.jp/monolish_benchmark_result/

# License
# 
