#!/bin/bash
echo "// This code is generated by gen_adds.sh
#include \"tensor_dense_adds.hpp\"
#include \"tensor_dense_line_adds.hpp\"

namespace monolish::blas {
"

# adds
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void adds(const $prec alpha, const $arg1 &A, $arg2 &C){ adds_core<$prec>(alpha, A, C);}"
    done
  done
done

echo ""

# adds_row
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void adds_row(const $arg1 &A, const $arg2 &x, $arg3 &C){ adds_row_core<$prec>(A, x, C);}"
      done
    done
  done
done

echo ""

# adds_col
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void adds_col(const $arg1 &A, const $arg2 &x, $arg3 &C){ adds_col_core<$prec>(A, x, C);}"
      done
    done
  done
done

echo ""

# adds_row line
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void adds_row(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C){ adds_row_core<$prec>(A, num, x, C);}"
      done
    done
  done
done

echo ""

# adds_col line
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void adds_col(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C){ adds_col_core<$prec>(A, num, x, C);}"
      done
    done
  done
done




echo "}"
