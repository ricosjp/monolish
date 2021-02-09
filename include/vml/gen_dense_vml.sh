C=const

echo " #pragma once
#include \"../common/monolish_common.hpp\"

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
/**
 * @brief
 * Vector and Matrix element-wise math library
 */
namespace vml {
"

## Dense matrix-matrix arithmetic
detail=(addition subtract multiplication division)
func=(add sub mul div)
for i in ${!detail[@]}; do
echo "
/**
 * @brief element by element ${detail[$i]} Dense matrix A and
 * Dense matrix B.
 * @param A monolish Dense Matrix (size M x N)
 * @param B monolish Dense Matrix (size M x N)
 * @param C monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
done

echo ""
################################################################

## Dense matrix-scalar arithmetic
detail=(addition subtract multiplication division)
func=(add sub mul div)
for i in ${!detail[@]}; do
echo "
/**
 * @brief element by element ${detail[$i]} Dense matrix A and
 * Dense matrix B.
 * @param A monolish Dense Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::Dense<$prec> &A, const $prec alpha, matrix::Dense<$prec> &C);"
done
done

echo ""
#############################################

## matrix-matrix pow
echo "
/**
 *@brief power to Dense matrix elements Dense matrix (C[0:N] = pow(A[0:N], B[0:N]))
 * @param A monolish Dense Matrix (size M x N)
 * @param B monolish Dense Matrix (size M x N)
 * @param C monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
for prec in double float; do
  echo "void pow(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
 
echo "
/**
 * @brief power to Dense matrix elements by scalar value (C[0:N] = pow(A[0:N], alpha))
 * @param A monolish Dense Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
for prec in double float; do
  echo "void pow(const matrix::Dense<$prec> &A, const $prec alpha, matrix::Dense<$prec> &C);"
done

echo ""
#############################################
## 2arg math
math=(sin sqrt sinh asin asinh tan tanh atan atanh ceil floor sign)
for math in ${math[@]}; do
echo "
/**
 * @brief $math to Dense matrix elements (C[0:nnz] = $math(A[0:nnz]))
 * @param A monolish Dense matrix (size M x N)
 * @param C monolish Dense matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
for prec in double float; do
  echo "void $math(const matrix::Dense<$prec> &A, matrix::Dense<$prec> &C);"
done
done

echo ""
#############################################

## matrix-matrix max min
detail=(greatest smallest)
func=(max min)
for i in ${!detail[@]}; do
echo "
/**
 * @brief Create a new Dense matrix with ${detail[$i]} elements of two matrices (C[0:nnz] = ${func[$i]}(A[0:nnz], B[0:nnz]))
 * @param A monolish Dense matrix (size M x N)
 * @param B monolish Dense matrix (size M x N)
 * @param C monolish Dense matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
done

echo ""

## Dense matrix max min
detail=(greatest smallest)
func=(max min)
for i in ${!detail[@]}; do
echo "
/**
 * @brief Finds the ${detail[$i]} element in Dense matrix (${func[$i]}(C[0:nnz]))
 * @param C monolish Dense matrix (size M x N)
 * @return ${detail[$i]} value
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
for prec in double float; do
    echo "$prec ${func[$i]}(const matrix::Dense<$prec> &C);"
done
done

echo ""
#############################################

## reciprocal
echo "
/**
 * @brief reciprocal to Dense matrix elements (C[0:nnz] = 1 / A[0:nnz])
 * @param A monolish Dense matrix (size M x N)
 * @param C monolish Dense matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
for prec in double float; do
  echo "void reciprocal(const matrix::Dense<$prec> &a, matrix::Dense<$prec> &y);"
done

echo "
} // namespace blas
} // namespace monolish "
