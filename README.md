# monolish: MONOlithic Liner equation Solvers for Highly-parallel architecture
monolish is どこでも動くmath kernel library

monolish 
- Works on any architecture 
- High performance(MPI, Multi-threading, SIMD)
- Communication I/F (GPU/MPI, send-recv
- Don’t reinvent the wheel (BLAS/SparseBLAS, Matrix Asm., Math func.,)
- Don't depend matrix structure for Linear Systems (Direct/Iterative, LA/Eigen)
- Build and Package Managing (cmake, Docker)
- Performance Logging to Find the bottleneck (logger)
- Continuous Integration／Continuous Delivery/Continuous Benchmark (monolish performance viewer)
- easy to implement other Language FFI (C++ and other language I/F)
- Don't depend Data type / type system (double/float, int/long, complex...)
- Don't implicitly allocate memory??
- Don't implement functions whose performance is unknown
- Don't require many code changes due to hardware changes (Intel, AMD, Power, ARM, A64fx, SXAT, NVIDIA GPU, AMD GPU...)
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

# Install
詳しくは [Doxygen](https://ricos.pages.ritc.jp/monolish/) を見てください
make, cmakeが使えます

## make
Install path is `$MONOLISH_DIR` 

### CPU
```
make cpu -j
make install
```

### GPU
```
make gpu -j
make install
```

## cmake

- -DBUILD\_GPU={ON/OFF}
- -DCMAKE\_INSTALL\_PREFIX=[dir]

# 開発者向け
monolishはchangelogに変更がない場合とclang-formatかけてない場合にCIでwarningがでる．

- clang formatはTOPDIRで make format するとgit addされてるファイル全部に自動でかかります
- changelogは `CHANGELOG.md` に主要な変更とマージリクエストの番号を書く

# MTG資料
初期設計資料 [GSlides](https://docs.google.com/presentation/d/16JvP7bTtxmfMP9hqflB7FVDrxueYxYa5U2PT-SkqB20/edit?usp=sharing)

Matrix Format [Gslides](https://docs.google.com/presentation/d/1wqyw9CmlHar84WxTgnoULn0_ZHZ7IxkUnLa_HkIwVQo/edit?usp=sharing)

ver1.0 MTG: [GSlides Link](https://docs.google.com/presentation/d/12LJXbFmAmKcEWtkIBCZm_klpqmAP6MIuvYCRAZnvwqQ/edit?usp=sharing)

# 関連URL

https://ricos.pages.ritc.jp/monolish_benchmark_result/

# License
# 
