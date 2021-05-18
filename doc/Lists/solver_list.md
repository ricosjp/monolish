# Implementation of Linear solvers{#solverlist}
monolish switches between `MKL`, `NVIDIA` and `OSS` at build time (see [here](@ref call_lib)).

This chapter explains what libraries are called by matrix and vector operations.

## Implementation of Linear Solvers
LAPACK is a complete direct solver for dense matrices. monolish calls LAPACK for direct solver for dense matrices.

The direct solver for sparse matrices is implemented in paradiso/mumps/cuSOLVER. monolish calls these libraries. Currently, only MKL and cuSOLVER is implemented.

Iterative solver for sparse matrix is implemented in MKL and CUDA libraries. However, the sparse matrix storage formats implemented by these libraries are different.
monolish does not use these libraries.
monolish has and provides an iterative solver implementation for sparse matrices.

In the future, we plan to implement a switch to call these libraries.


## Implementation status (solving / preconditioning) {#solverstatus}

|                                             | Dense / MKL   | Dense / NVIDIA | Dense / OSS   | Sparse / MKL  | Sparse / NVIDIA | Sparse / OSS  | LinearOperator   / MKL | LinearOperator / NVIDIA | LinearOperator / OSS |
|---------------------------------------------|---------------|----------------|---------------|---------------|-----------------|---------------|------------------------|-------------------------|----------------------|
| monolish::equation::LU                      | **Y** / N     | **Y** / N      | **Y** / N     | N / N         | N / N           | N / N         | N / N                  | N / N                   | N / N                |
| monolish::equation::QR                      | N / N         | N / N          | N / N         | N / N         | **Y** / N       | N / N         | N / N                  | N / N                   | N / N                |
| monolish::equation::Cholesky                | **Y** / N     | N / N          | **Y** / N     | N / N         | **Y** / N       | N / N         | N / N                  | N / N                   | N / N                |
| monolish::equation::CG                      | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N     | **Y** / N       | **Y** / N     | **Y** / N              | N / N                   | **Y** / N            |
| monolish::equation::BiCGSTAB                | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N     | **Y** / N       | **Y** / N     | **Y** / N              | N / N                   | **Y** / N            |
| monolish::equation::Jacobi                  | **Y** / **Y** | **Y** / **Y**  | **Y** / **Y** | **Y** / **Y** | **Y** / **Y**   | N / **Y**     | **Y** / **Y**          | N / N                   | **Y** / **Y**        |
| monolish::standard_eigen::LOBPCG            | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N     | **Y** / N       | **Y** / N     | **Y** / N              | N / N                   | **Y** / N            |
| monolish::standard_eigen::DC                | **Y** / N     | **Y** / N      | **Y** / N     | N / N         | N / N           | N / N         | N / N                  | N / N                   | N / N                |
| monolish::generalized_eigen::LOBPCG         | **Y** / N     | **Y** / N      | **Y** / N     | **Y** / N     | **Y** / N       | **Y** / N     | **Y** / N              | N / N                   | **Y** / N            |
| monolish::generalized_eigen::DC             | **Y** / N     | **Y** / N      | **Y** / N     | N / N         | N / N           | N / N         | N / N                  | N / N                   | N / N                |

# What libraries are called by solvers

## Direct Solvers for for Dense matrix
| func     | MKL                          | NVIDIA                    | OSS                                    |
|----------|------------------------------|---------------------------|--------------------------------------- |
| LU       | MKL                          | cuSOLVER                  | OpenBLAS+LAPACK                        |
| Cholesky | MKL                          | not impl.*                | OpenBLAS+LAPACK                        |
| QR       | todo) not impl.->MKL         | todo) not impl.->cuSOLVER | todo) not impl. -> OpenBLAS+LAPACK     |

- *) Cholesky is not impl. in cuSOLVER 11.1

## Direct Solvers for sparse matrix
| func     | MKL                            | NVIDIA           | OSS                             |
|----------|--------------------------------|------------------|-------------------------------- |
| LU       | todo) not impl.->MKL           | not impl.*       | todo) not impl. -> MUMPS        |
| Cholesky | todo) not impl.->MKL           | cuSOLVER         | todo) not impl. -> ????         |
| QR       | todo) not impl.->MKL           | cuSOLVER         | todo) not impl. -> ????         |

- *) sparse LU is not impl. in cuSOLVER 11.1

## Iterative solvers (for Dense, Sparse)

| func     | MKL            | NVIDIA         | OSS            |
|----------|----------------|----------------|----------------|
| CG       | monolish       | monolish       | monolish       |
| BiCGSTAB | monolish       | monolish       | monolish       |
| Jacobi   | monolish       | monolish       | monolish       |

## Preconditioners (for Dense, Sparse)

| func   | MKL            | NVIDIA   | OSS      |
|--------|----------------|----------|----------|
| Jacobi | monolish       | monolish | monolish |

# Standard Eigen Solvers

## For dense matrix

| func                     | MKL       | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| Devide and Conquer       | MKL       | cuSOLVER       | OpenBLAS+LAPACK |

## For sparse matrix

| func                     | MKL       | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| LOBPCG                   | monolish  | monolish       | monolish        |

# Generalized Eigen Solvers

## For dense matrix

| func                     | MKL       | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| Devide and Conquer       | MKL       | cuSOLVER       | OpenBLAS+LAPACK |


## For sparse matrix

| func                     | MKL       | NVIDIA         | OSS             |
|--------------------------|-----------|----------------|-----------------|
| LOBPCG                   | monolish  | monolish       | monolish        |
