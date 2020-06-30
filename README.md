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
第1回MTG [GSlides](https://docs.google.com/presentation/d/1LzTvWe_b_oKFHR2HP7gd1ds7nLxLUi2ncWVo9qk0x0c/edit?usp=sharing)

第2回MTG [GSlides](https://docs.google.com/presentation/d/1bgzDkHm5AHRyxxj2mM09zGMT9P9IkH21UNLrKanhyG0/edit?usp=sharing)
