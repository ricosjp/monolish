C=const

echo "
#pragma once
#include \"../common/monolish_common.hpp\"
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
"


## copy Dense
echo "
/**
* @brief double precision Dense matrix copy (C=A)
* @param A double precision monolish Dense matrix (size M x N)
* @param C double precision monolish Dense matrix (size M x N)
* @note
* - # of computation: M x N
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void copy(const matrix::Dense<$prec> &A, matrix::Dense<$prec> &C);"
done

echo ""

## copy LinearOperator
echo "
/**
* @brief double precision LinearOperator copy (C=A)
* @param A double precision monolish LinearOperator (size M x N)
* @param C double precision monolish LinearOperator (size M x N)
* @note
* - # of computation: M x N
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void copy(const matrix::LinearOperator<$prec> &A, matrix::LinearOperator<$prec> &C);"
done

echo ""

## copy CRS
echo "
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
*/ "
for prec in double float; do
  echo "void copy(const matrix::CRS<$prec> &A, matrix::CRS<$prec> &C);"
done

echo ""

##############################################

#mscal Dense
echo "
/**
* @brief double precision Densematrix scal: A = alpha * A
* @param alpha double precision scalar value
* @param A double precision Dense matrix (size M x N)
* @note
* - # of computation: MN
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void mscal(const $prec alpha, matrix::Dense<$prec> &A);"
done

echo ""

#mscal CRS
echo "
/**
* @brief double precision CRS matrix scal: A = alpha * A
* @param alpha double precision scalar value
* @param A double precision CRS matrix (size M x N)
* @note
* - # of computation: MN
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void mscal(const $prec alpha, matrix::CRS<$prec> &A);"
done

echo ""
##############################################

#matadd Dense
echo "
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
*/ "
for prec in double float; do
  echo "void matadd(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done

#matadd LinearOperator
echo "
/**
* @brief double precision LinearOperator addition: C = A + B
* @param A double precision LinearOperator (size M x N)
* @param B double precision LinearOperator (size M x N)
* @param C double precision LinearOperator (size M x N)
* @note
* - # of computation: 2 functions
* - Multi-threading: false
* - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matadd(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<> &C);"
done

#matadd CRS
echo "
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
*/ "
for prec in double float; do
  echo "void matadd(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done

echo ""

#matsub Dense
echo "
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
*/ "
for prec in double float; do
  echo "void matsub(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done

#matsub LinearOperator
echo "
/**
* @brief double precision LinearOperator subtract: C = A - B
* @param A double precision LinearOperator (size M x N)
* @param B double precision LinearOperator (size M x N)
* @param C double precision LinearOperator (size M x N)
* @note
* - # of computation: 2 functions
* - Multi-threading: false
* - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matsub(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<$prec> &C);"
done

#matsub CRS
echo "
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
*/ "
for prec in double float; do
  echo "void matsub(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done

#################################
echo ""

#matsub Dense
echo "
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
*/ "
for prec in double float; do
  echo "void matmul(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done

#matsub LinearOperator
echo "
/**
* @brief double precision LinearOperator multiplication: C = AB
* @param A double precision LinearOperator (size M x K)
* @param B double precision LinearOperator (size K x N)
* @param C double precision LinearOperator (size M x N)
* @note
* - # of computation: 2 functions
* - Multi-threading: false
* - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matmul(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<$prec> &C);"
done

#matsub CRS
echo "
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
*/ "
for prec in double float; do
  echo "void matmul(const matrix::CRS<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done


echo "
} // namespace blas
} // namespace monolish
"
