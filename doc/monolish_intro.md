# What's is monolish? {#monolish_intro}

## Introduction

monolish is a linear solver library that monolithically fuses variable data type, matrix structures, matrix data format, vender specific data transfer APIs, and vender specific numerical algebra libraries.

monolish is a vendor-independent open source library written in C++ that aims to be grand unified linear algebra library on any hardware.

BLAS has 150+ functions and lapack has 1000+ functions. These are complete software for dense matrix operations. 
However, these are not enough due to the growing needs of users. 
In particular, sparse matrix operations and machine learning kernels are not implemented. 

To solve this, the libraries (MKL, cusolver, etc.) by each hardware vendor extended functions.
Sadly, these software APIs are not unified. These are language or vecder specific.


BLAS is a complete library in terms of performance, but the function names depend on the data type.
Python numpy, Julia, matlab, etc. define APIs that eliminate these dependencies and call BLAS in them.
Fast Sparse BLAS is implemented in MKL and CUDA Libraries.
Sadly, there is no open source Sparse BLAS library that works on all hardware.
However, due to the needs of application users, the Python and Julia libraries have implemented Sparse BLAS on their own.  The performance of these functions is not clear.

Most of the HPC engineers use C/C++/Fortran, while most of the application engineers use Python and Julia.
We think it is necessary to develop C/C++/Fortran libraries for dense and sparse matrices, which are the base of numerical computation.

monolish is a monolithic C++ numerical library designed to bridge the gap between HPC engineers and application engineers.
monolish provides an API that integrates the numerical algebra libraries of each vendor.
monolish calls the vendor-developed numerical algebra libraries whenever possible.
monolish implements and provides functions that are not implemented in these libraries.

We dream that monolish will be used as a backend for python, julia, etc. libraries in the future.

monolish solves cumbersome package management by Docker.

monolish:
- written in C++14
- Provide GPU acceleration using OpenMP Offloading
- Provice BLAS / Sparse BLAS / VML / Dense direct solvers / sparse iterative solvers

## Development policy for high performance 

monolish has five development policies.

### 1. Don't require users to change programs due to changes in data types

Override functions and do not implement datatype-dependent functions.

In the future, imaginary numbers and high precision data types will be supported.

### 2. Don't require users to change their programs due to changes in hardware architecture.

monolish integrates vendor-implemented BLAS libraries and device communication APIs for each architecture.
It provides a new BLAS/LAPACK/Sparse BLAS that monolithically integrates types, matrix formats, and vendor-specific APIs.

To support AMD GPUs and Intel Xe, the internal device programs are implemented using OpenMP Offloading.

### 3. Don't require users to change their programs due to changes in sparse matrix format.

In monolish, all matrix formats, including Dense matrix, are defined as Sparse matrix format.
The same interface is designed for all classes to minimize program changes due to storage format changes.

### 4. Don't implement functions that clearly do not provide performance.

It is important to have the same functionality in all classes.
However, there are often operations that cannot be made faster in principle.
For example, operations on columns of a matrix in CRS format.

If a useful function is implemented, many users will use it even if its performance is low.
In monolish, even if there is a bias in functionality between classes, functions that are not fast enough in principle are not implemented.

We guarantee, as much as possible, that programs implemented with a combination of monolish functions will run faster.


### 5. Don't allocate memory in ways that users cannot anticipate.

Compound functions and operator overloading are useful.
However, they often require the function to allocate memory for the return vector or matrix internally.
In monolish, we do not implement functions that allocate memory in a way that is not intuitively expected by the user.
