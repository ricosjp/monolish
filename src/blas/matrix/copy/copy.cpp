// This code is generated by gen_copy.sh
#include "dense_copy.hpp"
#include "crs_copy.hpp"
#include "linearOperator_copy.hpp"

namespace monolish::blas {

void copy(const matrix::Dense<double> &A, matrix::Dense<double> &C){ copy_core(A, C); }
void copy(const matrix::Dense<double> &A, view_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const matrix::Dense<double> &A, view_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const matrix::Dense<double> &A, view_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<double>,double> &A, matrix::Dense<double> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<double>,double> &A, view_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<double>,double> &A, view_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<double>,double> &A, view_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<double>,double> &A, matrix::Dense<double> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<double>,double> &A, view_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<double>,double> &A, view_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<double>,double> &A, view_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<double>,double> &A, matrix::Dense<double> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<double>,double> &A, view_Dense<vector<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<double>,double> &A, view_Dense<matrix::Dense<double>,double> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<double>,double> &A, view_Dense<tensor::tensor_Dense<double>,double> &C){ copy_core(A, C); }
void copy(const matrix::Dense<float> &A, matrix::Dense<float> &C){ copy_core(A, C); }
void copy(const matrix::Dense<float> &A, view_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const matrix::Dense<float> &A, view_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const matrix::Dense<float> &A, view_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<float>,float> &A, matrix::Dense<float> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<float>,float> &A, view_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<float>,float> &A, view_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<vector<float>,float> &A, view_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<float>,float> &A, matrix::Dense<float> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<float>,float> &A, view_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<float>,float> &A, view_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<matrix::Dense<float>,float> &A, view_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<float>,float> &A, matrix::Dense<float> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<float>,float> &A, view_Dense<vector<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<float>,float> &A, view_Dense<matrix::Dense<float>,float> &C){ copy_core(A, C); }
void copy(const view_Dense<tensor::tensor_Dense<float>,float> &A, view_Dense<tensor::tensor_Dense<float>,float> &C){ copy_core(A, C); }

void copy(const matrix::CRS<double> &A, matrix::CRS<double> &C){ copy_core(A, C); }
void copy(const matrix::CRS<float> &A, matrix::CRS<float> &C){ copy_core(A, C); }

void copy(const matrix::LinearOperator<double> &A, matrix::LinearOperator<double> &C){ copy_core(A, C); }
void copy(const matrix::LinearOperator<float> &A, matrix::LinearOperator<float> &C){ copy_core(A, C); }
}
