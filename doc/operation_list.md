# Implementation of matrix/vector operations{#oplist}
This section describes the implementation of each function.
The first goal of monolish is to implement the basic operations that allow the BLAS, Sparse BLAS, and VML functions of the MKL and CUDA libraries to work on all hardware environments.

On intel CPUs and NVIDIA GPUs, MKL and CUDA libraries are the fastest.
The monolish uses these libraries as much as possible, and implements the missing functions by itself.
When compiling, it switches the function to be called if the MKL or CUDA libraries are available or not. Switch dependency libraries at compile time.

The branch for the case where MKL or CUDA libraries are not available is called `OSS`.
`OSS` is an implementation for architectures such as AMD, ARM, Power, etc.

In `OSS`, we assume that only CBLAS compatible BLAS libraries and LAPACK can be used.
The functions of MKL and CUDA libraries that are not implemented in CBLAS are implemented in monolish.

![](img/call_blas.png)

## Implementation of Linear Solvers
LAPACK is a complete direct solver for dense matrices. monolish calls LAPACK for direct solver for dense matrices.

The direct solver for sparse matrices is implemented in paradiso/mumps/cusolver. monolish calls these libraries. Currently, only cusolver is implemented.

Iterative solver for sparse matrix is implemented in MKL and CUDA libraries. However, the sparse matrix storage formats implemented by these libraries are different.
monolish has and provides an iterative solver implementation for sparse matrices.

In the future, if MKL or CUDA libraries are available, we plan to implement a switch to call these libraries.

# BLAS 

## BLAS Lv1

| func  | Intel    | NVIDIA   | OSS       |
|-------|----------|----------|-----------|
| copy  | MKL      | cuBLAS   | CBLAS     |
| sum   | monolish | monolish | monolish  |
| asum  | MKL      | cuBLAS   | CBLAS     |
| axpy  | MKL      | cuBLAS   | CBLAS     |
| axpyz | monolish | monolish | monolish  |
| xpay  | monolish | monolish | monolish  |
| dot   | MKL      | cuBLAS   | CBLAS     |
| nrm1  | monolish | monolish | monolish  |
| nrm2  | MKL      | cuBLAS   | CBLAS     |
| scal  | MKL      | cuBLAS   | CBLAS     |

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
| Dense | MKL           | cuBLAS   | CBLAS     |
| CRS   | MKL           | cuSparse | monolish  |

## BLAS Lv3 (matmul)

| func        | Intel         | NVIDIA             | OSS           |
|-------------|---------------|--------------------|---------------|
| Dense-Dense | MKL           | cuBLAS             | CBLAS         |
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
| copy  | MKL           | cuBLAS           | CBLAS    |

## vector helper functions of VML

| func        | Intel         | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS    |

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
| copy        | MKL           | cuBLAS         | CBLAS    |
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
| copy        | MKL           | cuBLAS         | CBLAS    |

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
