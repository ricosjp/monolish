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
## vector-vector arithmetic
detail=(addition subtract multiplication division)
func=(add sub mul div)
for i in ${!detail[@]}; do
echo "
/**
 * @brief element by element ${detail[$i]} of vector a and vector b.
 * @param a monolish vector (size N)
 * @param b monolish vector (size N)
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
  for prec in double float; do
    for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
        for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
          echo "void ${func[$i]}(const $arg1 &a, const $arg2 &b, $arg3 &y);"
        done
      done
    done
  done
done

echo ""

## scalar-vector arithmetic
for i in ${!detail[@]}; do
echo "
/**
 * @brief element by element ${detail[$i]} of vector a and vector b.
 * @param a monolish vector (size N)
 * @param alpha scalar value
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
  for prec in double float; do
    for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
        echo "void ${func[$i]}(const $arg1 &a, const $prec alpha, $arg2 &y);"
      done
    done
  done
done

echo ""
#############################################

## vector-vector pow
echo "
/**
 * @brief power to vector elements by double precision vector
 *(y[0:N] = pow(a[0:N], b[0]:N]))
* @param a monolish vector (size N)
* @param b monolish vector (size N)
* @param y monolish vector (size N)
* @note
* - # of computation: N
* - Multi-threading: true
* - GPU acceleration: true
*    - # of data transfer: 0
*/ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
        echo "void pow(const $arg1 &a, const $arg2 &b, $arg3 &y);"
      done
    done
  done
done

echo ""

## scalar-vector pow
echo "
/**
 * @brief power to vector elements by double precision scalar
 *value (y[0:N] = pow(a[0:N], alpha))
 * @param a monolish vector (size N)
 * @param alpha scalar value
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void pow(const $arg1 &a, const $prec alpha, $arg2 &y);"
    done
  done
done

echo ""
#############################################


## 2arg math
math=(sin sqrt sinh asin asinh tan tanh atan atanh ceil floor sign)
for math in ${math[@]}; do
echo "
/**
 * @brief $math to vector elements (y[0:N] = $math(a[0:N]))
 * @param a monolish vector (size N)
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
  for prec in double float; do
    for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
        echo "void $math(const $arg1 &a, $arg2 &y);"
      done
    done
  done
done

echo ""
#############################################

## vector-vector max min
detail=(greatest smallest)
func=(max min)
for i in ${!detail[@]}; do
echo "
/**
 * @brief Create a new vector with ${detail[$i]} elements of two matrices (y[0:N] = ${func[$i]}(a[0:N], b[0:N]))
 * @param a monolish vector (size N)
 * @param b monolish vector (size N)
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
        echo "void ${func[$i]}(const $arg1 &a, const $arg2 &b, $arg3 &y);"
      done
    done
  done
done
done

echo ""

## vector max min
detail=(greatest smallest)
func=(max min)
for i in ${!detail[@]}; do
echo "
/**
 * @brief Finds the ${detail[$i]} element in vector (${func[$i]}(y[0:N]))
 * @param y monolish vector (size N)
 * @return ${detail[$i]} value
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
*/"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "$prec ${func[$i]}(const $arg1 &y);"
  done
done
done

echo ""
#############################################

## reciprocal
math=reciprocal
echo "
/**
 * @brief reciprocal to double precision vector elements (y[0:N] = 1 / a[0:N])
 * @param a monolish vector (size N)
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
*/ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void $math(const $arg1 &a, $arg2 &y);"
    done
  done
done

echo "
} // namespace blas
} // namespace monolish "
