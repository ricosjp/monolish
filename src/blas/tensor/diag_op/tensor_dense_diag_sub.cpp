// Tshi code is generated by gen_dense_diag_sub.sh
#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "tensor_dense_diag_op.hpp"

namespace monolish {

template <>
void monolish::tensor::tensor_Dense<double>::diag_sub(const double alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::view_tensor_Dense<vector<double>, double>::diag_sub(
    const double alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<double>, double>::diag_sub(
    const double alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<double>,
                                 double>::diag_sub(const double alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::tensor::tensor_Dense<float>::diag_sub(const float alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::view_tensor_Dense<vector<float>, float>::diag_sub(
    const float alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<float>, float>::diag_sub(
    const float alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<float>, float>::diag_sub(
    const float alpha) {
  tensor_Dense_diag_sub_core(*this, alpha);
}

template <>
void monolish::tensor::tensor_Dense<double>::diag_sub(
    const vector<double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<double>::diag_sub(
    const view1D<vector<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<double>::diag_sub(
    const view1D<matrix::Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<double>::diag_sub(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<double>, double>::diag_sub(
    const vector<double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<double>, double>::diag_sub(
    const view1D<vector<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<double>, double>::diag_sub(
    const view1D<matrix::Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<double>, double>::diag_sub(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<double>, double>::diag_sub(
    const vector<double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<double>, double>::diag_sub(
    const view1D<vector<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<double>, double>::diag_sub(
    const view1D<matrix::Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<double>, double>::diag_sub(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<double>,
                                 double>::diag_sub(const vector<double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<double>, double>::
    diag_sub(const view1D<vector<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<double>, double>::
    diag_sub(const view1D<matrix::Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<double>, double>::
    diag_sub(const view1D<tensor::tensor_Dense<double>, double> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<float>::diag_sub(const vector<float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<float>::diag_sub(
    const view1D<vector<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<float>::diag_sub(
    const view1D<matrix::Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::tensor::tensor_Dense<float>::diag_sub(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<float>, float>::diag_sub(
    const vector<float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<float>, float>::diag_sub(
    const view1D<vector<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<float>, float>::diag_sub(
    const view1D<matrix::Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<vector<float>, float>::diag_sub(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<float>, float>::diag_sub(
    const vector<float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<float>, float>::diag_sub(
    const view1D<vector<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<float>, float>::diag_sub(
    const view1D<matrix::Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<matrix::Dense<float>, float>::diag_sub(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<float>, float>::diag_sub(
    const vector<float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<float>, float>::diag_sub(
    const view1D<vector<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<float>, float>::diag_sub(
    const view1D<matrix::Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
template <>
void monolish::view_tensor_Dense<tensor::tensor_Dense<float>, float>::diag_sub(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  tensor_Dense_diag_sub_core(*this, vec.size(), vec.begin());
}
} // namespace monolish
