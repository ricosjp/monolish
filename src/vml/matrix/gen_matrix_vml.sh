#!/bin/bash
echo "//this code is generated by gen_matrix_vml.sh
#include \"../../../include/monolish_vml.hpp\"
#include \"../../internal/monolish_internal.hpp\"
#include \"./arithmetic/matadd.hpp\"
#include \"./arithmetic/matsub.hpp\"
#include \"./arithmetic/matmul.hpp\"
#include \"./arithmetic/matdiv.hpp\"
#include \"./math_2op/matasin.hpp\"
#include \"./math_2op/matasinh.hpp\"
#include \"./math_2op/matatan.hpp\"
#include \"./math_2op/matatanh.hpp\"
#include \"./math_2op/matfloor.hpp\"
#include \"./math_2op/matreciprocal.hpp\"
#include \"./math_2op/matsign.hpp\"
#include \"./math_2op/matsin.hpp\"
#include \"./math_2op/matsinh.hpp\"
#include \"./math_2op/matsqrt.hpp\"
#include \"./math_2op/matceil.hpp\"
#include \"./math_2op/mattan.hpp\"
#include \"./math_2op/mattanh.hpp\"
#include \"./math_2op/matexp.hpp\"
#include \"./math_2op/matalo.hpp\"
#include \"./math_1_3op/matmax.hpp\"
#include \"./math_1_3op/matmin.hpp\"
#include \"./math_1_3op/matpow.hpp\"

namespace monolish {
namespace vml {
"

## $MAT matrix-matrix arithmetic
funcs=(add sub mul div)
for func in ${funcs[@]}; do
  for prec in double float; do
    for arg1 in matrix::CRS\<$prec\>; do
      echo "void ${func}(const $arg1 &A, const $arg1 &B, $arg1 &C){mm${func}_core(A, B, C);}"
    done
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void ${func}(const $arg1 &A, const $arg2 &B, $arg3 &C){mm${func}_core(A, B, C);}"
        done
      done
    done
  done
done

echo ""
################################################################

## $MAT matrix-scalar arithmetic
funcs=(add sub mul div)
for func in ${funcs[@]}; do
  for prec in double float; do
    for arg1 in matrix::CRS\<$prec\>; do
      echo "void ${func}(const $arg1 &A, const $prec alpha, $arg1 &C){sm${func}_core(A, alpha, C);}"
    done
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void ${func}(const $arg1 &A, const $prec alpha, $arg2 &C){sm${func}_core(A, alpha, C);}"
      done
    done
  done
done

echo ""
#############################################

## matrix-matrix pow
for prec in double float; do
  for arg1 in matrix::CRS\<$prec\>; do
    echo "void pow(const $arg1 &A, const $arg1 &B, $arg1 &C){mmpow_core(A, B, C);}"
  done
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void pow(const $arg1 &A, const $arg2 &B, $arg3 &C){mmpow_core(A, B, C);}"
      done
    done
  done
done

## matrix-scalar pow
for prec in double float; do
  for arg1 in matrix::CRS\<$prec\>; do
    echo "void pow(const $arg1 &A, const $prec alpha, $arg1 &C){smpow_core(A, alpha, C);}"
  done
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void pow(const $arg1 &A, const $prec alpha, $arg2 &C){smpow_core(A, alpha, C);}"
    done
  done
done

echo ""
#############################################

## matrix alo
for prec in double float; do
  for arg1 in matrix::CRS\<$prec\>; do
    echo "void alo(const $arg1 &A, const $prec alpha, const $prec beta, $arg1 &C){malo_core(A, alpha, beta, C);}"
  done
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void alo(const $arg1 &A, const $prec alpha, const $prec beta, $arg2 &C){malo_core(A, alpha, beta, C);}"
    done
  done
done

echo ""
#############################################
## 2arg math
math=(sin sqrt sinh asin asinh tan tanh atan atanh ceil floor sign reciprocal exp)
for math in ${math[@]}; do
  for prec in double float; do
    for arg1 in matrix::CRS\<$prec\>; do
      echo "void $math(const $arg1 &A, $arg1 &C){m${math}_core(A, C);}"
    done
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void $math(const $arg1 &A, $arg2 &C){m${math}_core(A, C);}"
      done
    done
  done
done

echo ""
#############################################

## matrix-matrix max min
detail=(greatest smallest)
funcs=(max min)
for func in ${funcs[@]}; do
  for prec in double float; do
    for arg1 in matrix::CRS\<$prec\>; do
      echo "void ${func}(const $arg1 &A, const $arg1 &B, $arg1 &C){mm${func}_core(A, B, C);}"
    done
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void ${func}(const $arg1 &A, const $arg2 &B, $arg3 &C){mm${func}_core(A, B, C);}"
        done
      done
    done
  done
done

echo ""

## matrix-scalar max min
detail=(greatest smallest)
funcs=(max min)
for func in ${funcs[@]}; do
  for prec in double float; do
    for arg1 in matrix::CRS\<$prec\>; do
      echo "void ${func}(const $arg1 &A, const $prec alpha, $arg1 &C){sm${func}_core(A, alpha, C);}"
    done
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void ${func}(const $arg1 &A, const $prec alpha, $arg2 &C){sm${func}_core(A, alpha, C);}"
      done
    done
  done
done

echo ""

## $MAT matrix max min
detail=(greatest smallest)
funcs=(max min)
for func in ${funcs[@]}; do
  for prec in double float; do
    for arg1 in matrix::CRS\<$prec\>; do
      echo "$prec ${func}(const $arg1 &C){return m${func}_core<$arg1,$prec>(C);}"
    done
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "$prec ${func}(const $arg1 &C){return m${func}_core<$arg1,$prec>(C);}"
    done
  done
done

echo ""
#############################################

echo "
} // namespace vml
} // namespace monolish "
