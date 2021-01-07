#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

template <typename MATRIX, typename T> T equation::solver<MATRIX, T>::get_residual(vector<T> &x) {
  switch (resid_method) {
  case 0:
    return blas::nrm2(x);
    break;
  default:
    throw std::runtime_error("error vector size is not same");
    break;
  }
}

template double equation::solver<matrix::CRS<double>, double>::get_residual(vector<double> &x);
template float equation::solver<matrix::CRS<float>, float>::get_residual(vector<float> &x);
} // namespace monolish
