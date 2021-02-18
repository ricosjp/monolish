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
| CRS   | MKL           | cuSparse | monolish  |

## BLAS Lv3 (matmul)

| func        | Intel         | NVIDIA             | OSS           |
|-------------|---------------|--------------------|---------------|
| Dense-Dense | MKL           | cuBLAS             | CBLAS互換     |
| CRS-Dense   | MKL           | monolish           | monolish(AVX) |

- Todo) support CRS-Dense SpMM by NVIDIA cusparse (Rowmajor SpMM need cuda 11.x)

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

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
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
| reciprocal   |       monolish |       monolish |       monolish |
| max(v)       |       MKL      |       monolish |       monolish |
| max(v,v)     |       MKL      |       monolish |       monolish |
| min(v)       |       MKL      |       monolish |       monolish |
| min(v,v)     |       MKL      |       monolish |       monolish |

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

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
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
| reciprocal   |       monolish |       monolish |       monolish |
| max(v)       |       MKL      |       monolish |       monolish |
| max(v,v)     |       MKL      |       monolish |       monolish |
| min(v)       |       MKL      |       monolish |       monolish |
| min(v,v)     |       MKL      |       monolish |       monolish |

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

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
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
| reciprocal   |       monolish |       monolish |       monolish |
| max(v)       |       MKL      |       monolish |       monolish |
| max(v,v)     |       MKL      |       monolish |       monolish |
| min(v)       |       MKL      |       monolish |       monolish |
| min(v,v)     |       MKL      |       monolish |       monolish |

## Row/Col vector and scalar operations (CRS)

| func | Intel            | NVIDIA            | OSS             |
|------|------------------|-------------------|-----------------|
| add  | todo) not impl.  | todo) not impl.   | todo) not impl. |
| sub  | todo) not impl.  | todo) not impl.   | todo) not impl. |
| mul  | todo) not impl.  | todo) not impl.   | todo) not impl. |
| div  | todo) not impl.  | todo) not impl.   | todo) not impl. |

## Row/Col vector and vector operations (CRS)

| func | Intel            | NVIDIA            | OSS             |
|------|------------------|-------------------|-----------------|
| add  | todo) not impl.  | todo) not impl.   | todo) not impl. |
| sub  | todo) not impl.  | todo) not impl.   | todo) not impl. |
| mul  | todo) not impl.  | todo) not impl.   | todo) not impl. |
| div  | todo) not impl.  | todo) not impl.   | todo) not impl. |

# Linear Solvers

## Dense (あまり決まってない)
| func     | Intel                                        | NVIDIA                    | OSS             |
|----------|----------------------------------------------|---------------------------|---------------- |
| LU       | MKL                                          | cusolver                  | OpenBLAS+LAPACK |
| Cholesky | MKL                                          | todo) not impl.*          | OpenBLAS+LAPACK |
| QR       | todo) not impl.->MKL                         | todo) not impl.->cusolver |                 |

- *) Cholesky is not impl. in cusolver11.1

## Sparse LA

| func     | Intel          | NVIDIA         | OSS            |
|----------|----------------|----------------|----------------|
| CG       | monolish       | monolish       | monolish       |
| BiCGSTAB | monolish       | monolish       | monolish       |
| Jacobi   | monolish       | monolish       | monolish       |

## Sparse LA Preconditioner

| func   | Intel                                | NVIDIA   | OSS      |
|--------|--------------------------------------|----------|----------|
| Jacobi | monolish                             | monolish | monolish |

## Sparse Eigen

| func    | Intel                     | NVIDIA                    | OSS                       |
|---------|---------------------------|---------------------------|---------------------------|
| Lanczos | todo) not impl.->monolish | todo) not impl.->monolish | todo) not impl.->monolish |
| Arnoldi | todo) not impl.->monolish | todo) not impl.->monolish | todo) not impl.->monolish |
