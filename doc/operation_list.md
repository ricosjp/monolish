# Operation list {#oplist_md}
![](img/call_blas.png)


# Introduction
- ここに説明を書く
- vectorにはview1Dも突っ込める(ようにする)

# BLAS 

## BLAS Lv1

| func  | Intel    | NVIDIA   | OSS       |
|-------|----------|----------|-----------|
| copy  | MKL      | cuBLAS   | CBLAS互換 |
| sum   | monolish | monolish | monolish  |
| asum  | MKL      | cuBLAS   | CBLAS互換 |
| axpy  | MKL      | cuBLAS   | CBLAS互換 |
| axpyz | monolish | monolish | monolish  |
| xpay  | monolish | monolish | monolish  |
| dot   | MKL      | cuBLAS   | CBLAS互換 |
| nrm1  | monolish | monolish | monolish  |
| nrm2  | MKL      | cuBLAS   | CBLAS互換 |
| scal  | MKL      | cuBLAS   | CBLAS互換 |

## Extended BLAS Lv1

| func                | Intel    | NVIDIA   | OSS      |
|---------------------|----------|----------|----------|
| matrix scale(Dense) | monolish | monolish | monolish |
| matrix scale(CRS)   | monolish | monolish | monolish |
| vecadd              | monolish | monolish | monolish |
| vecsub              | monolish | monolish | monolish |

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

# Vector (and view1D) Operations

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

# Linear Solvers

## Direct Solvers for for Dense matrix
| func     | Intel                                        | NVIDIA                    | OSS                                    |
|----------|----------------------------------------------|---------------------------|--------------------------------------- |
| LU       | MKL                                          | cusolver                  | OpenBLAS+LAPACK                        |
| Cholesky | MKL                                          | not impl.*                | OpenBLAS+LAPACK                        |
| QR       | todo) not impl.->MKL                         | todo) not impl.->cusolver | todo) not impl. -> OpenBLAS+LAPACK     |

- *) Cholesky is not impl. in cusolver 11.1

## Direct Solvers for sparse matrix
| func     | Intel                          | NVIDIA           | OSS                             |
|----------|--------------------------------|------------------|-------------------------------- |
| LU       | todo) not impl.->MKL           | not impl.*       | todo) not impl. -> MUMPS        |
| Cholesky | todo) not impl.->MKL           | cusolver         | todo) not impl. -> ????         |
| QR       | todo) not impl.->MKL           | cusolver         | todo) not impl. -> ????         |

- *) sparse LU is not impl. in cusolver 11.1

## Iterative solvers (only CRS now, We will support Dense)

| func     | Intel          | NVIDIA         | OSS            |
|----------|----------------|----------------|----------------|
| CG       | monolish       | monolish       | monolish       |
| BiCGSTAB | monolish       | monolish       | monolish       |
| Jacobi   | monolish       | monolish       | monolish       |

## Preconditioners of Sparse Linear solver

| func   | Intel          | NVIDIA   | OSS      |
|--------|----------------|----------|----------|
| Jacobi | monolish       | monolish | monolish |

# Standard Eigen Solvers

## For dense matrix

| func                     | Intel     | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| Devide and Conquer       | MKL       | cusolver       | OpenBLAS+LAPACK |

## For sparse matrix

| func                     | Intel     | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| LOBPCG                   | monolish  | monolish       | monolish        |

# Generalized Eigen Solvers

## For dense matrix

| func                     | Intel     | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| Devide and Conquer       | MKL       | cusolver       | OpenBLAS+LAPACK |


## For sparse matrix

| func                     | Intel     | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| LOBPCG                   | monolish  | monolish       | monolish        |
