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
## vecadd
echo "
/**
 * @brief element by element addition of vector a and vector b.
 * @param a monolish vector (size N)
 * @param b monolish vector (size N)
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
echo "void vecadd(const $arg1 &a, const $arg2 &b, $arg3 &y);"
      done
    done
  done
done

## vecsub
echo "
/**
 * @brief element by element subtract of vector a and vector b.
 * @param a monolish vector (size N)
 * @param b monolish vector (size N)
 * @param y monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
echo "void vecsub(const $arg1 &a, const $arg2 &b, $arg3 &y);"
      done
    done
  done
done


## copy
echo "
/**
 * @brief double precision vector\<$prec\> copy (y=a)
 * @param a double precision monolish vector\<$prec\> (size N)
 * @param y double precision monolish vector\<$prec\> (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void copy($C $arg1 &x, $arg2 &y);"
    done
  done
done

echo ""

## asum
echo "
/**
 * @brief double precision vector\<$prec\> asum (absolute sum)
 * @param x double precision monolish vector\<$prec\> (size N)
 * @return The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "void asum($C $arg1 &x, $prec &ans);"
  done
done

echo ""

## asum
echo "
/**
 * @brief double precision vector\<$prec\> asum (absolute sum)
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param ans The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "$prec asum($C $arg1 &x);"
  done
done

## sum
echo "
/**
 * @brief double precision vector\<$prec\> sum
 * @param x double precision monolish vector\<$prec\> (size N)
 * @return The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "void sum($C $arg1 &x, $prec &ans);"
  done
done

echo ""

## sum
echo "
/**
 * @brief double precision vector\<$prec\> sum
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param ans The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "$prec sum($C $arg1 &x);"
  done
done

echo ""

## axpy
echo "
/**
 * @brief double precision axpy: y = ax + y
 * @param alpha double precision scalar value
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param y double precision monolish vector\<$prec\> (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void axpy(const $prec alpha, const $arg1 &x, $arg2 &y);"
    done
  done
done

echo ""

## axpyz
echo "
/**
 * @brief double precision axpyz: z = ax + y
 * @param alpha double precision scalar value
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param y double precision monolish vector\<$prec\> (size N)
 * @param z double precision monolish vector\<$prec\> (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
        echo "void axpyz(const $prec alpha, const $arg1 &x, const $arg2 &y, $arg3 &z);"
      done
    done
  done
done

echo ""

## dot
echo "
/**
 * @brief double precision inner product (dot)
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param y double precision monolish vector\<$prec\> (size N)
 * @return The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void dot($C $arg1 &x, $C $arg2 &y, $prec &ans);"
    done
  done
done

echo ""

## dot
echo "
/**
 * @brief double precision inner product (dot)
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param y double precision monolish vector\<$prec\> (size N)
 * @param ans The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "$prec dot($C $arg1 &x, $C vector<$prec> &y);"
    done
  done
done

echo ""

## nrm1
echo "
/**
 * @brief double precision nrm1: sum(abs(x[0:N]))
 * @param x double precision monolish vector\<$prec\> (size N)
 * @return The result of the nrm1
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "void nrm1($C $arg1 &x, $prec &ans);"
  done
done

echo ""

## nrm1
echo "
/**
 * @brief double precision nrm1: sum(abs(x[0:N]))
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param ans The result of the nrm1
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "$prec nrm1($C $arg1 &x);"
  done
done

echo ""

## nrm2
echo "
/**
 * @brief double precision nrm2: ||x||_2
 * @param x double precision monolish vector\<$prec\> (size N)
 * @return The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "void nrm2($C $arg1 &x, $prec &ans);"
  done
done

echo ""

## nrm2
echo "
/**
 * @brief double precision nrm2: ||x||_2
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param ans The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    echo "$prec nrm2($C $arg1 &x);"
  done
done

echo ""

## scal
echo "
/**
 * @brief double precision scal: x = alpha * x
 * @param alpha double precision scalar value
 * @param x double precision monolish vector\<$prec\> (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void scal(const $prec alpha, $arg1 &x);"
  done
done

echo ""

## xpay
echo "
/**
 * @brief double precision xpay: y = x + ay
 * @param alpha double precision scalar value
 * @param x double precision monolish vector\<$prec\> (size N)
 * @param y double precision monolish vector\<$prec\> (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
"
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\>; do
      echo "void xpay(const $prec alpha, const $arg1 &x, $arg2 &y);"
    done
  done
done

echo "
} // namespace blas
} // namespace monolish
"
