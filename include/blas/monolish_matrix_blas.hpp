
#pragma once
#include "../common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
/**
* @brief
* Basic Linear Algebra Subprograms for Dense Matrix, Sparse Matrix, Vector and
* Scalar
*/
namespace blas {


/**
* @brief double precision Dense matrix copy (C=A)
* @param A double precision monolish Dense matrix (size M x N)
* @param C double precision monolish Dense matrix (size M x N)
* @note
* - # of computation: M x N
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void copy(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void copy(const matrix::Dense<float> &A, matrix::Dense<float> &C);


/**
* @brief double precision LinearOperator copy (C=A)
* @param A double precision monolish LinearOperator (size M x N)
* @param C double precision monolish LinearOperator (size M x N)
* @note
* - # of computation: M x N
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void copy(const matrix::LinearOperator<double> &A, matrix::LinearOperator<double> &C);
void copy(const matrix::LinearOperator<float> &A, matrix::LinearOperator<float> &C);


/**
* @brief double precision CRS matrix copy (y=a)
* @param A double precision monolish CRS matrix (size M x N)
* @param C double precision monolish CRS matrix (size M x N)
* @note
* - # of computation: M x N
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
* @warning
* A and C must be same non-zero structure
*/ 
void copy(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void copy(const matrix::CRS<float> &A, matrix::CRS<float> &C);


/**
* @brief double precision Densematrix scal: A = alpha * A
* @param alpha double precision scalar value
* @param A double precision Dense matrix (size M x N)
* @note
* - # of computation: MN
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void mscal(const double alpha, matrix::Dense<double> &A);
void mscal(const float alpha, matrix::Dense<float> &A);


/**
* @brief double precision CRS matrix scal: A = alpha * A
* @param alpha double precision scalar value
* @param A double precision CRS matrix (size M x N)
* @note
* - # of computation: MN
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void mscal(const double alpha, matrix::CRS<double> &A);
void mscal(const float alpha, matrix::CRS<float> &A);


/**
* @brief double precision Dense matrix addition: C = A + B
* @param A double precision Dense matrix (size M x N)
* @param B double precision Dense matrix (size M x N)
* @param C double precision Dense matrix (size M x N)
* @note
* - # of computation: MN
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void matadd(const matrix::Dense<double> &A, const matrix::Dense<double> &B, matrix::Dense<double> &C);
void matadd(const matrix::Dense<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C);

/**
* @brief double precision LinearOperator addition: C = A + B
* @param A double precision LinearOperator (size M x N)
* @param B double precision LinearOperator (size M x N)
* @param C double precision LinearOperator (size M x N)
* @note
* - # of computation: 2 functions
* - Multi-threading: false
* - GPU acceleration: false
*/ 
void matadd(const matrix::LinearOperator<double> &A, const matrix::LinearOperator<double> &B, matrix::LinearOperator<> &C);
void matadd(const matrix::LinearOperator<float> &A, const matrix::LinearOperator<float> &B, matrix::LinearOperator<> &C);

/**
* @brief double precision CRS matrix addition: C = A + B (A and B must be
* same non-zero structure)
* @param A double precision CRS matrix (size M x N)
* @param B double precision CRS matrix (size M x N)
* @param C double precision CRS matrix (size M x N)
* @note
* - # of computation: nnz
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
* @warning
* A and B must be same non-zero structure
*/ 
void matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B, matrix::CRS<double> &C);
void matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B, matrix::CRS<float> &C);


/**
* @brief double precision Dense matrix subtract: C = A - B
* @param A double precision Dense matrix (size M x N)
* @param B double precision Dense matrix (size M x N)
* @param C double precision Dense matrix (size M x N)
* @note
* - # of computation: MN
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void matsub(const matrix::Dense<double> &A, const matrix::Dense<double> &B, matrix::Dense<double> &C);
void matsub(const matrix::Dense<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C);

/**
* @brief double precision LinearOperator subtract: C = A - B
* @param A double precision LinearOperator (size M x N)
* @param B double precision LinearOperator (size M x N)
* @param C double precision LinearOperator (size M x N)
* @note
* - # of computation: 2 functions
* - Multi-threading: false
* - GPU acceleration: false
*/ 
void matsub(const matrix::LinearOperator<double> &A, const matrix::LinearOperator<double> &B, matrix::LinearOperator<double> &C);
void matsub(const matrix::LinearOperator<float> &A, const matrix::LinearOperator<float> &B, matrix::LinearOperator<float> &C);

/**
* @brief double precision CRS matrix subtract: C = A - B (A and B must be
* same non-zero structure)
* @param A double precision CRS matrix (size M x N)
* @param B double precision CRS matrix (size M x N)
* @param C double precision CRS matrix (size M x N)
* @note
* - # of computation: nnz
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
* @warning
* A B must be same non-zero structure
*/ 
void matsub(const matrix::CRS<double> &A, const matrix::CRS<double> &B, matrix::CRS<double> &C);
void matsub(const matrix::CRS<float> &A, const matrix::CRS<float> &B, matrix::CRS<float> &C);


/**
* @brief double precision Dense matrix multiplication: C = AB
* @param A double precision Dense matrix (size M x K)
* @param B double precision Dense matrix (size K x N)
* @param C double precision Dense matrix (size M x N)
* @note
* - # of computation: 2MNK
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void matmul(const matrix::Dense<double> &A, const matrix::Dense<double> &B, matrix::Dense<double> &C);
void matmul(const matrix::Dense<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C);

/**
* @brief double precision LinearOperator multiplication: C = AB
* @param A double precision LinearOperator (size M x K)
* @param B double precision LinearOperator (size K x N)
* @param C double precision LinearOperator (size M x N)
* @note
* - # of computation: 2 functions
* - Multi-threading: false
* - GPU acceleration: false
*/ 
void matmul(const matrix::LinearOperator<double> &A, const matrix::LinearOperator<double> &B, matrix::LinearOperator<double> &C);
void matmul(const matrix::LinearOperator<float> &A, const matrix::LinearOperator<float> &B, matrix::LinearOperator<float> &C);

/**
* @brief double precision CRS and Dense matrix multiplication: C = AB
* @param A double precision CRS matrix (size M x K)
* @param B double precision Dense matrix (size K x N)
* @param C double precision Dense matrix (size M x N)
* @note
* - # of computation: 2*N*nnz
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ 
void matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B, matrix::Dense<double> &C);
void matmul(const matrix::CRS<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C);

} // namespace blas
} // namespace monolish

