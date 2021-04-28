# Implementation of matrix/vector operations{#oplist}
monolish switches between `MKL`, `NVIDIA` and `OSS` at build time (see [here](@ref call_lib)).

This chapter explains what libraries are called by matrix and vector operations.

# BLAS 

## BLAS Lv1

| func  | MKL      | NVIDIA   | OSS       |
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

| func                | MKL      | NVIDIA   | OSS      |
|---------------------|----------|----------|----------|
| matrix scale(Dense) | monolish | monolish | monolish |
| matrix scale(CRS)   | monolish | monolish | monolish |
| vecadd              | monolish | monolish | monolish |
| vecsub              | monolish | monolish | monolish |

## BLAS Lv2 (matvec)

| func  | MKL           | NVIDIA   | OSS       |
|-------|---------------|----------|-----------|
| Dense | MKL           | cuBLAS   | CBLAS     |
| CRS   | MKL           | cuSPARSE | monolish  |

## BLAS Lv3 (matmul)

| func        | MKL           | NVIDIA             | OSS           |
|-------------|---------------|--------------------|---------------|
| Dense-Dense | MKL           | cuBLAS             | CBLAS         |
| CRS-Dense   | MKL           | monolish           | monolish(AVX) |

- Todo) support CRS-Dense SpMM by NVIDIA cusparse (Rowmajor SpMM need cuda 11.x)

## Extended BLAS Lv3: Matrix add/sub (C=A+B)

| func  | MKL      | NVIDIA   | OSS      |
|-------|----------|----------|----------|
| Dense | MKL      | monolish | monolish |
| CRS   | MKL      | monolish | monolish |

# Vector (and view1D) Operations

## scalar-vector VML

| func  | MKL           | NVIDIA           | OSS       |
|-------|---------------|------------------|-----------|
| add   | monolish      | monolish         | monolish  |
| sub   | monolish      | monolish         | monolish  |
| mul   | monolish      | monolish         | monolish  |
| div   | monolish      | monolish         | monolish  |

## vector-vector VML

| func  | MKL           | NVIDIA           | OSS      |
|-------|---------------|------------------|----------|
| add   | MKL           | monolish         | monolish |
| sub   | MKL           | monolish         | monolish |
| mul   | MKL           | monolish         | monolish |
| div   | MKL           | monolish         | monolish |
| equal | monolish      | monolish         | monolish |
| copy  | MKL           | cuBLAS           | CBLAS    |

## vector helper functions of VML

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS    |

## vector Mathematical functions of VML

| func         | MKL            | NVIDIA         | OSS            |
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

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## Dense-Dense operations of VML

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## Dense helper functions of VML

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS    |
| transpose   | monolish      | monolish       | monolish |

## Dense mathematical functions of VML

| func         | MKL            | NVIDIA         | OSS            |
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

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## CRS-Dense operations of VML

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| add         | MKL           | monolish       | monolish |
| sub         | MKL           | monolish       | monolish |
| mul         | MKL           | monolish       | monolish |
| div         | MKL           | monolish       | monolish |

## CRS helper functions of VML

| func        | MKL           | NVIDIA         | OSS      |
|-------------|---------------|----------------|----------|
| equal       | monolish      | monolish       | monolish |
| not equal   | monolish      | monolish       | monolish |
| copy        | MKL           | cuBLAS         | CBLAS    |

## CRS mathematical functions of VML

| func         | MKL            | NVIDIA         | OSS            |
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
