# Implementation of Linear solvers{#solverlist}
monolish switches between `Intel`, `NVIDIA` and `OSS` at build time (see [here](@ref call_lib)).

This chapter explains what libraries are called by matrix and vector operations.

## Implementation of Linear Solvers
LAPACK is a complete direct solver for dense matrices. monolish calls LAPACK for direct solver for dense matrices.

The direct solver for sparse matrices is implemented in paradiso/mumps/cusolver. monolish calls these libraries. Currently, only MKL and cusolver is implemented.

Iterative solver for sparse matrix is implemented in MKL and CUDA libraries. However, the sparse matrix storage formats implemented by these libraries are different.
monolish does not use these libraries.
monolish has and provides an iterative solver implementation for sparse matrices.

In the future, we plan to implement a switch to call these libraries.


## Implementation status (solving / preconditioning) {#solverstatus}
|                                                                          | Dense, Intel | Dense, NVIDIA | Dense, OSS   | Sparse, Intel | Sparse, NVIDIA | Sparse, OSS  |
|--------------------------------------------------------------------------|---------------|----------------|---------------|----------------|-----------------|---------------|
| @ref monolish::equation::LU   "LU"                                       | **Y** / N     | **Y** / N      | **Y** / N     | N / N          | N / N           | N / N         |
| @ref monolish::equation::QR   "QR"                                       | N / N         | N / N          | N / N         | N / N          | **Y** / N       | N / N         |
| @ref monolish::equation::Cholesky   "Cholesky"                           | **Y** / N     | N / N          | **Y** / N     | N / N          | **Y** / N       | N / N         |
| @ref monolish::equation::CG   "CG"                                       | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N      | **Y** / N       | **Y** / N     |
| @ref monolish::equation::BiCGSTAB   "BiCGSTAB"                           | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N      | **Y** / N       | **Y** / N     |
| @ref monolish::equation::Jacobi   "Jacobi"                               | **Y** / **Y** | **Y** / **Y**  | **Y** / **Y** | **Y** / **Y**  | **Y** / **Y**   | **Y** / **Y** |
| @ref monolish::equation::none   "none"                                   | **Y** / **Y** | **Y** / **Y**  | **Y** / **Y** | **Y** / **Y**  | **Y** / **Y**   | **Y** / **Y** |
| @ref monolish::standard_eigen::LOBPCG   "LOBPCG (standard_eigen)"        | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N      | **Y** / N       | **Y** / N     |
| @ref monolish::standard_eigen::DC   "DC (standard_eigen)"                | **Y** / N     | **Y** / N      | **Y** / N     | N / N          | N / N           | N / N         |
| @ref monolish::generalized_eigen::LOBPCG   "LOBPCG (generalized_eigen)"  | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N      | **Y** / N       | **Y** / N     |
| @ref monolish::generalized_eigen::DC   "DC (generalized_eigen)"          | **Y** / N     | **Y** / N      | **Y** / N     | N / N          | N / N           | N / N         |

# what libraries are called by solvers

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
