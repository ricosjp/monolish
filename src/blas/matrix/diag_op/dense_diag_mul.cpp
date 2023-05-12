#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "dense_diag_op.hpp"

namespace monolish::matrix {

// mul scalar
template <> void monolish::matrix::Dense<double>::diag_mul(const double alpha) {
  Dense_diag_mul_core(*this, alpha);
};
template <> void monolish::matrix::Dense<float>::diag_mul(const float alpha) {
  Dense_diag_mul_core(*this, alpha);
};

// mul vector
template <>
void monolish::matrix::Dense<double>::diag_mul(const vector<double> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::Dense<float>::diag_mul(const vector<float> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};

template <>
void monolish::matrix::Dense<double>::diag_mul(
    const view1D<vector<double>, double> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::Dense<float>::diag_mul(
    const view1D<vector<float>, float> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};

template <>
void monolish::matrix::Dense<double>::diag_mul(
    const view1D<matrix::Dense<double>, double> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::Dense<float>::diag_mul(
    const view1D<matrix::Dense<float>, float> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};

template <>
void monolish::matrix::Dense<double>::diag_mul(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::Dense<float>::diag_mul(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  Dense_diag_mul_core(*this, vec.size(), vec.begin());
};
} // namespace monolish::matrix
