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

## LinearOperator matrix-matrix arithmetic
detail=(addition subtract)
func=(add sub)
for i in ${!detail[@]}; do
echo "
/**
 * @brief element by element ${detail[$i]} LinearOperator matrix A and
 * LinearOperator matrix B.
 * @param A monolish LinearOperator Matrix (size M x N)
 * @param B monolish LinearOperator Matrix (size M x N)
 * @param C monolish LinearOperator Matrix (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<$prec> &C);"
done
done

echo ""
################################################################

## LinearOperator matrix-scalar arithmetic
detail=(addition subtract multiplication division)
func=(add sub mul div)
for i in ${!detail[@]}; do
echo "
/**
 * @brief element by element ${detail[$i]} LinearOperator matrix A and
 * LinearOperator matrix B.
 * @param A monolish LinearOperator Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish LinearOperator Matrix (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::LinearOperator<$prec> &A, const $prec &alpha, matrix::LinearOperator<$prec> &C);"
done
done

echo "
} // namespace blas
} // namespace monolish "
