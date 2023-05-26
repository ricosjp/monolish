#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "crs_diag_op.hpp"

namespace monolish::matrix {

// sub scalar

template <> void monolish::matrix::CRS<double>::diag_sub(const double alpha) {
  CRS_diag_sub_core(*this, alpha);
}
template <> void monolish::matrix::CRS<float>::diag_sub(const float alpha) {
  CRS_diag_sub_core(*this, alpha);
};

// sub vector

template <>
void monolish::matrix::CRS<double>::diag_sub(const vector<double> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::CRS<float>::diag_sub(const vector<float> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};

template <>
void monolish::matrix::CRS<double>::diag_sub(
    const view1D<vector<double>, double> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::CRS<float>::diag_sub(
    const view1D<vector<float>, float> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};

template <>
void monolish::matrix::CRS<double>::diag_sub(
    const view1D<matrix::Dense<double>, double> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::CRS<float>::diag_sub(
    const view1D<matrix::Dense<float>, float> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};

template <>
void monolish::matrix::CRS<double>::diag_sub(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};
template <>
void monolish::matrix::CRS<float>::diag_sub(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  CRS_diag_sub_core(*this, vec.size(), vec.begin());
};
} // namespace monolish::matrix
