# Operation list {#oplist_md}
![](img/call_blas.png)
# BLAS Operations

## BLAS Lv1

| func  | Intel    | NVIDIA   | OSS       |
|-------|----------|----------|-----------|
| asum  | MKL      | cuBLAS   | CBLAS互換 |
| axpy  | MKL      | cuBLAS   | CBLAS互換 |
| axpyz | monolish | monolish | monolish  |
| dot   | MKL      | cuBLAS   | CBLAS互換 |
| nrm2  | MKL      | cuBLAS   | CBLAS互換 |
| scal  | MKL      | cuBLAS   | CBLAS互換 |
| xpay  | monolish | monolish | monolish  |

## Extended BLAS Lv1: Matrix scale (alpha\*A)

| func  | Intel    | NVIDIA   | OSS      |
|-------|----------|----------|----------|
| Dense | monolish | monolish | monolish |
| CRS   | monolish | monolish | monolish |

## BLAS Lv2 (matvec)

| func  | Intel         | NVIDIA   | OSS       |
|-------|---------------|----------|-----------|
| Dense | MKL           | cuBLAS   | CBLAS互換 |
| CRS   | monolish->MKL | cuSparse | monolish  |

## BLAS Lv3 (matmul)

| func        | Intel         | NVIDIA             | OSS           |
|-------------|---------------|--------------------|---------------|
| Dense-Dense | MKL           | cuBLAS             | CBLAS互換     |
| CRS-Dense   | monolish->MKL | monolish->cuSparse | monolish(AVX) |

# Vector Operations

## Arithmetics (scalar-vector)

| func  | Intel         | NVIDIA           | OSS       |
|-------|---------------|------------------|-----------|
| add   | monolish      | monolish         | monolish  |
| sub   | monolish      | monolish         | monolish  |
| mul   | monolish      | monolish         | monolish  |
| div   | monolish      | monolish         | monolish  |

## Arithmetics (vector-vector)

| func  | Intel         | NVIDIA           | OSS      |
|-------|---------------|------------------|----------|
| add   | MKL           | monolish         | monolish |
| sub   | MKL           | monolish         | monolish |
| mul   | MKL           | monolish         | monolish |
| div   | monolish      | monolish         | monolish |
| equal | monolish      | monolish         | monolish |
| copy  | MKL           | cuBLAS           | CBLAS互換|

## Mathematical functions (vector)

| func | Intel     | NVIDIA         | OSS            |
|------|-----------|----------------|----------------|
| sin  | none->MKL | none->monolish | none->monolish |
| cos  | none->MKL | none->monolish | none->monolish |
| tan  | none->MKL | none->monolish | none->monolish |
| sinh | none->MKL | none->monolish | none->monolish |
| cosh | none->MKL | none->monolish | none->monolish |
| tanh | none->MKL | none->monolish | none->monolish |
| exp  | none->MKL | none->monolish | none->monolish |
| log  | none->MKL | none->monolish | none->monolish |

# Dense Matrix Operations

## Arithmetics (Dense)

| func        | Intel         | NVIDIA         | OSS            |
|-------------|---------------|----------------|----------------|
| add         | MKL           | monolish       | monolish       |
| sub         | none->MKL     | none->monolish | none->monolish |
| equal       | none->MKL     | none->monolish | none->monolish |
| not   equal | none->MKL     | none->monolish | none->monolish |
| copy        | MKL           | cuBLAS         | CBLAS互換      |
| transpose   | monolish      | monolish       | monolish       |

## Row/Col vector and scalar operations (Dense)

| func | Intel    | NVIDIA   | OSS      |
|------|----------|----------|----------|
| add  | monolish | monolish | monolish |
| sub  | monolish | monolish | monolish |
| mul  | monolish | monolish | monolish |
| div  | monolish | monolish | monolish |

## Row/Col vector and vector operations (Dense)

| func | Intel    | NVIDIA   | OSS      |
|------|----------|----------|----------|
| add  | monolish | monolish | monolish |
| sub  | monolish | monolish | monolish |
| mul  | monolish | monolish | monolish |
| div  | monolish | monolish | monolish |

## Mathematical functions (Dense)

| func | Intel     | NVIDIA         | OSS            |
|------|-----------|----------------|----------------|
| sin  | none->MKL | none->monolish | none->monolish |
| cos  | none->MKL | none->monolish | none->monolish |
| tan  | none->MKL | none->monolish | none->monolish |
| sinh | none->MKL | none->monolish | none->monolish |
| cosh | none->MKL | none->monolish | none->monolish |
| tanh | MKL       | none->monolish | none->monolish |
| exp  | none->MKL | none->monolish | none->monolish |
| log  | none->MKL | none->monolish | none->monolish |

# CRS Matrix Operations

## Arithmetics (CRS)

| func        | Intel         | NVIDIA         | OSS            |
|-------------|---------------|----------------|----------------|
| add         | MKL           | monolish       | monolish       |
| sub         | none->MKL     | none           | none           |
| equal       | none->MKL     | none->monolish | none->monolish |
| not   equal | none->MKL     | none->monolish | none->monolish |
| copy        | MKL           | cublas         | CBLAS互換      |
| transpose   | none          | none           | none           |

## Row/Col vector and scalar operations (CRS)

| func | Intel | NVIDIA | OSS  |
|------|-------|--------|------|
| add  | none  | none   | none |
| sub  | none  | none   | none |
| mul  | none  | none   | none |
| div  | none  | none   | none |

## Row/Col vector and vector operations (CRS)

| func | Intel | NVIDIA | OSS  |
|------|-------|--------|------|
| add  | none  | none   | none |
| sub  | none  | none   | none |
| mul  | none  | none   | none |
| div  | none  | none   | none |

## Mathematical functions (CRS)

| func | Intel     | NVIDIA         | OSS            |
|------|-----------|----------------|----------------|
| sin  | none->MKL | none->monolish | none->monolish |
| cos  | none->MKL | none->monolish | none->monolish |
| tan  | none->MKL | none->monolish | none->monolish |
| sinh | none->MKL | none->monolish | none->monolish |
| cosh | none->MKL | none->monolish | none->monolish |
| tanh | none->MKL | none->monolish | none->monolish |
| exp  | none->MKL | none->monolish | none->monolish |
| log  | none->MKL | none->monolish | none->monolish |

# Linear Solvers

## Dense (あまり決まってない)
| func     | Intel                                        | NVIDIA         | OSS |
|----------|----------------------------------------------|----------------|-----|
| LU       | none->MKL                                    | none->cusolver | ??? |
| QR       | none->MKL                                    | none->cusolver | ??? |
| Cholesky | none->MKL                                    | none->cusolver | ??? |
| ???      | 全部やってるとLAPACKラッパーになっちゃう．． |                |     |

## Sparse LA

| func     | Intel          | NVIDIA         | OSS            |
|----------|----------------|----------------|----------------|
| CG       | monolish       | monolish       | monolish       |
| BiCGSTAB | none->monolish | none->monolish | none->monolish |
| Jacobi   | none->monolish | none->monolish | none->monolish |

## Sparse LA Preconditioner

| func   | Intel                                | NVIDIA   | OSS      |
|--------|--------------------------------------|----------|----------|
| Jacobi | monolish                             | monolish | monolish |
| ???    | Jacobiだけじゃ弱すぎるのでなにか．． |          |          |

## Sparse Eigen

| func    | Intel          | NVIDIA         | OSS            |
|---------|----------------|----------------|----------------|
| Lanczos | none->monolish | none->monolish | none->monolish |
| Arnoldi | none->monolish | none->monolish | none->monolish |
