C=const

echo "#pragma once
#include \"asum.hpp\"
#include \"axpy.hpp\"
#include \"axpyz.hpp\"
#include \"copy.hpp\"
#include \"dot.hpp\"
#include \"nrm1.hpp\"
#include \"nrm2.hpp\"
#include \"scal.hpp\"
#include \"sum.hpp\"
#include \"vecadd.hpp\"
#include \"vecsub.hpp\"
#include \"xpay.hpp\"

namespace monolish {
namespace blas {
"

## vecadd
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
        if [ $prec = "double" ]
        then
          echo "void vecadd(const $arg1 &a, const $arg2 &b, $arg3 &y){vecadd_core(a, b, y);}"
        else
          echo "void vecadd(const $arg1 &a, const $arg2 &b, $arg3 &y){vecadd_core(a, b, y);}"
        fi
      done
    done
  done
done

echo ""

## vecsub
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
      for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
        if [ $prec = "double" ]
        then
          echo "void vecsub(const $arg1 &a, const $arg2 &b, $arg3 &y){vecsub_core(a, b, y);}"
        else
          echo "void vecsub(const $arg1 &a, const $arg2 &b, $arg3 &y){vecsub_core(a, b, y);}"
        fi
      done
    done
  done
done
# 
# 
# ## copy
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       echo "void copy($C $arg1 &x, $arg2 &y);"
#     done
#   done
# done
# 
# echo ""
# 
# ## asum
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "void asum($C $arg1 &x, $prec &ans);"
#   done
# done
# 
# echo ""
# 
# ## asum
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "$prec asum($C $arg1 &x);"
#   done
# done
# 
# ## sum
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "void sum($C $arg1 &x, $prec &ans);"
#   done
# done
# 
# echo ""
# 
# ## sum
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "$prec sum($C $arg1 &x);"
#   done
# done
# 
# echo ""
# 
# ## axpy
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       echo "void axpy(const $prec alpha, const $arg1 &x, $arg2 &y);"
#     done
#   done
# done
# 
# echo ""
# 
# ## axpyz
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       for arg3 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#         echo "void axpyz(const $prec alpha, const $arg1 &x, const $arg2 &y, $arg3 &z);"
#       done
#     done
#   done
# done
# 
# echo ""
# 
# ## dot
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       echo "void dot($C $arg1 &x, $C $arg2 &y, $prec &ans);"
#     done
#   done
# done
# 
# echo ""
# 
# ## dot
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       echo "$prec dot($C $arg1 &x, $C $arg2 &y);"
#     done
#   done
# done
# 
# echo ""
# 
# ## nrm1
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "void nrm1($C $arg1 &x, $prec &ans);"
#   done
# done
# 
# echo ""
# 
# ## nrm1
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "$prec nrm1($C $arg1 &x);"
#   done
# done
# 
# echo ""
# 
# ## nrm2
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "void nrm2($C $arg1 &x, $prec &ans);"
#   done
# done
# 
# echo ""
# 
# ## nrm2
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     echo "$prec nrm2($C $arg1 &x);"
#   done
# done
# 
# echo ""
# 
# ## scal
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       echo "void scal(const $prec alpha, $arg1 &x);"
#   done
# done
# 
# echo ""
# 
# ## xpay
# for prec in double float; do
#   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\>; do
#       echo "void xpay(const $prec alpha, const $arg1 &x, $arg2 &y);"
#     done
#   done
# done
# 
echo "
} // namespace blas
} // namespace monolish"
