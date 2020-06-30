# MONOlish (MONOlithic Liner equation Solvers for Highly-parallel architecture)

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

# MTG資料
初期設計資料 [GSlides](https://docs.google.com/presentation/d/16JvP7bTtxmfMP9hqflB7FVDrxueYxYa5U2PT-SkqB20/edit?usp=sharing)

Matrix Format [Gslides](https://docs.google.com/presentation/d/1wqyw9CmlHar84WxTgnoULn0_ZHZ7IxkUnLa_HkIwVQo/edit?usp=sharing)
