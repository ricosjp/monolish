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
| sum   | monolish | monolish | monolish  |

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

## Extended BLAS Lv3: Matrix add/sub (C=A+B)

| func  | Intel    | NVIDIA   | OSS      |
|-------|----------|----------|----------|
| Dense | MKL      | monolish | monolish |
| CRS   | MKL      | monolish | monolish |

# Vector Operations

## scalar-vector VML

| func  | Intel         | NVIDIA           | OSS       |
|-------|---------------|------------------|-----------|
| add   | monolish      | monolish         | monolish  |
| sub   | monolish      | monolish         | monolish  |
| mul   | monolish      | monolish         | monolish  |
| div   | monolish      | monolish         | monolish  |

## vector-vector VML

| func  | Intel         | NVIDIA           | OSS      |
|-------|---------------|------------------|----------|
| add   | MKL           | monolish         | monolish |
| sub   | MKL           | monolish         | monolish |
| mul   | MKL           | monolish         | monolish |
| div   | MKL           | monolish         | monolish |
| equal | monolish      | monolish         | monolish |
| copy  | MKL           | cuBLAS           | CBLAS互換|

## vector helper functions of VML
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS互換|
| transpose   | monolish      | monolish       | monolish |

## vector Mathematical functions of VML

| func         | Intel          | NVIDIA         | OSS            |
|--------------|----------------|----------------|----------------|
| sin          |       MKL      |       monolish |       monolish |
| sinh         |       MKL      |       monolish |       monolish |
| arcsin       |       MKL      |       monolish |       monolish |
| arcsinh      |       MKL      |       monolish |       monolish |
| tan          |       MKL      |       monolish |       monolish |
| tanh         |       MKL      |       monolish |       monolish |
| arctan       |       MKL      |       monolish |       monolish |
| arctanh      |       MKL      |       monolish |       monolish |
| power(v,v)   |       MKL      |       monolish |       monolish |
| power(v,s)   |       monolish |       monolish |       monolish |
| sqrt         |       MKL      |       monolish |       monolish |
| ceil         |       MKL      |       monolish |       monolish |
| floor        |       MKL      |       monolish |       monolish |
| sign         |       monolish |       monolish |       monolish |
| max(v)       | none->MKL      | none->monolish | none->monolish |
| max(v,v)     | none->MKL      | none->monolish | none->monolish |
| min(v)       | none->MKL      | none->monolish | none->monolish |
| min(v,v)     | none->MKL      | none->monolish | none->monolish |

# Dense Matrix Operations

## scalar-Dense operations of VML

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## Dense-Dense operations of VML

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## Dense helper functions of VML
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS互換|
| transpose   | monolish      | monolish       | monolish |

## Dense mathematical functions of VML

| func         | Intel          | NVIDIA         | OSS            |
|--------------|----------------|----------------|----------------|
| sin          |       MKL      |       monolish |       monolish |
| sinh         |       MKL      |       monolish |       monolish |
| arcsin       |       MKL      |       monolish |       monolish |
| arcsinh      |       MKL      |       monolish |       monolish |
| tan          |       MKL      |       monolish |       monolish |
| tanh         |       MKL      |       monolish |       monolish |
| arctan       |       MKL      |       monolish |       monolish |
| arctanh      |       MKL      |       monolish |       monolish |
| power(v,v)   |       MKL      |       monolish |       monolish |
| power(v,s)   |       monolish |       monolish |       monolish |
| sqrt         |       MKL      |       monolish |       monolish |
| ceil         |       MKL      |       monolish |       monolish |
| floor        |       MKL      |       monolish |       monolish |
| sign         |       monolish |       monolish |       monolish |
| max(v)       | none->MKL      | none->monolish | none->monolish |
| max(v,v)     | none->MKL      | none->monolish | none->monolish |
| min(v)       | none->MKL      | none->monolish | none->monolish |
| min(v,v)     | none->MKL      | none->monolish | none->monolish |

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

# CRS Matrix Operations

## scalar-CRS operations of VML

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## CRS-Dense operations of VML

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## CRS helper functions of VML
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS互換|
| transpose   | monolish      | monolish       | monolish |

## CRS mathematical functions of VML

| func         | Intel          | NVIDIA         | OSS            |
|--------------|----------------|----------------|----------------|
| sin          |       MKL      |       monolish |       monolish |
| sinh         |       MKL      |       monolish |       monolish |
| arcsin       |       MKL      |       monolish |       monolish |
| arcsinh      |       MKL      |       monolish |       monolish |
| tan          |       MKL      |       monolish |       monolish |
| tanh         |       MKL      |       monolish |       monolish |
| arctan       |       MKL      |       monolish |       monolish |
| arctanh      |       MKL      |       monolish |       monolish |
| power(v,v)   |       MKL      |       monolish |       monolish |
| power(v,s)   |       monolish |       monolish |       monolish |
| sqrt         |       MKL      |       monolish |       monolish |
| ceil         |       MKL      |       monolish |       monolish |
| floor        |       MKL      |       monolish |       monolish |
| sign         |       monolish |       monolish |       monolish |
| max(v)       | none->MKL      | none->monolish | none->monolish |
| max(v,v)     | none->MKL      | none->monolish | none->monolish |
| min(v)       | none->MKL      | none->monolish | none->monolish |
| min(v,v)     | none->MKL      | none->monolish | none->monolish |

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
