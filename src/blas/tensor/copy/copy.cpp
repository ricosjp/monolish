// This code is generated by gen_copy.sh
#include "tensor_dense_copy.hpp"

namespace monolish::blas {

void copy(const tensor::tensor_Dense<double> &A, tensor::tensor_Dense<double> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<double> &A, view_tensor_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<double> &A, view_tensor_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<double> &A, view_tensor_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<double>,double> &A, tensor::tensor_Dense<double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<double>,double> &A, view_tensor_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<double>,double> &A, view_tensor_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<double>,double> &A, view_tensor_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<double>,double> &A, tensor::tensor_Dense<double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<double>,double> &A, view_tensor_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<double>,double> &A, view_tensor_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<double>,double> &A, view_tensor_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<double>,double> &A, tensor::tensor_Dense<double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<double>,double> &A, view_tensor_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<double>,double> &A, view_tensor_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<double>,double> &A, view_tensor_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<float> &A, view_tensor_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<float> &A, view_tensor_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const tensor::tensor_Dense<float> &A, view_tensor_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<float>,float> &A, tensor::tensor_Dense<float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<float>,float> &A, view_tensor_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<float>,float> &A, view_tensor_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<vector<float>,float> &A, view_tensor_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<float>,float> &A, tensor::tensor_Dense<float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<float>,float> &A, view_tensor_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<float>,float> &A, view_tensor_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<matrix::Dense<float>,float> &A, view_tensor_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<float>,float> &A, tensor::tensor_Dense<float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<float>,float> &A, view_tensor_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<float>,float> &A, view_tensor_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_tensor_Dense<tensor::tensor_Dense<float>,float> &A, view_tensor_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
}
