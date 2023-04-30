#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "crs_diag_op.hpp"

namespace monolish::matrix {

// div scalar
template <> void monolish::matrix::CRS<double>::diag_div(const double alpha) {
  CRS_diag_div_core(*this, alpha);
};
template <> void monolish::matrix::CRS<float>::diag_div(const float alpha) {
  CRS_diag_div_core(*this, alpha);
};

// div vector
template <>
void monolish::matrix::CRS<double>::diag_div(const vector<double> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};
template <>
void monolish::matrix::CRS<float>::diag_div(const vector<float> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};

template <>
void monolish::matrix::CRS<double>::diag_div(
    const view1D<vector<double>, double> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};
template <>
void monolish::matrix::CRS<float>::diag_div(
    const view1D<vector<float>, float> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};

template <>
void monolish::matrix::CRS<double>::diag_div(
    const view1D<matrix::Dense<double>, double> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};
template <>
void monolish::matrix::CRS<float>::diag_div(
    const view1D<matrix::Dense<float>, float> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};

template <>
void monolish::matrix::CRS<double>::diag_div(
    const view1D<tensor::tensor_Dense<double>, double> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};
template <>
void monolish::matrix::CRS<float>::diag_div(
    const view1D<tensor::tensor_Dense<float>, float> &vec) {
  CRS_diag_div_core(*this, vec.size(), vec.data());
};
} // namespace monolish::matrix
