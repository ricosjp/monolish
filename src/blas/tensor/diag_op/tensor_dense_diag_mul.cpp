#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "tensor_dense_diag_op.hpp"

namespace monolish::tensor {

template <>
void monolish::tensor::tensor_Dense<double>::diag_mul(const double alpha) {
  tensor_Dense_diag_mul_core(*this, alpha);
};
template <>
void monolish::tensor::tensor_Dense<float>::diag_mul(const float alpha) {
  tensor_Dense_diag_mul_core(*this, alpha);
};

template <>
void monolish::tensor::tensor_Dense<double>::diag_mul(
    const vector<double> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};
template <>
void monolish::tensor::tensor_Dense<float>::diag_mul(const vector<float> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};

template <>
void monolish::tensor::tensor_Dense<double>::diag_mul(
    const view1D<vector<double>, double> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};
template <>
void monolish::tensor::tensor_Dense<float>::diag_mul(
    const view1D<vector<float>, float> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};

template <>
void monolish::tensor::tensor_Dense<double>::diag_mul(
    const view1D<matrix::Dense<double>, double> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};
template <>
void monolish::tensor::tensor_Dense<float>::diag_mul(
    const view1D<matrix::Dense<float>, float> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};

template <>
void monolish::tensor::tensor_Dense<double>::diag_mul(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};
template <>
void monolish::tensor::tensor_Dense<float>::diag_mul(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  tensor_Dense_diag_mul_core(*this, vec.size(), vec.data());
};
} // namespace monolish::tensor
