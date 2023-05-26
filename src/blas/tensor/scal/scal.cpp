// This code is generated by gen_scal.sh
#include "tensor_dense_scal.hpp"

namespace monolish::blas {

void tscal(const double alpha, tensor::tensor_Dense<double> &A) {
  tscal_core(alpha, A);
}
void tscal(const double alpha, view_tensor_Dense<vector<double>, double> &A) {
  tscal_core(alpha, A);
}
void tscal(const double alpha,
           view_tensor_Dense<matrix::Dense<double>, double> &A) {
  tscal_core(alpha, A);
}
void tscal(const double alpha,
           view_tensor_Dense<tensor::tensor_Dense<double>, double> &A) {
  tscal_core(alpha, A);
}
void tscal(const float alpha, tensor::tensor_Dense<float> &A) {
  tscal_core(alpha, A);
}
void tscal(const float alpha, view_tensor_Dense<vector<float>, float> &A) {
  tscal_core(alpha, A);
}
void tscal(const float alpha,
           view_tensor_Dense<matrix::Dense<float>, float> &A) {
  tscal_core(alpha, A);
}
void tscal(const float alpha,
           view_tensor_Dense<tensor::tensor_Dense<float>, float> &A) {
  tscal_core(alpha, A);
}
} // namespace monolish::blas
