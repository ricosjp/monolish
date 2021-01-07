#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

template <typename T> T solver::solver<T>::get_residual(vector<T> &x) {
  switch (resid_method) {
  case 0:
    return blas::nrm2(x);
    break;
  default:
    throw std::runtime_error("error vector size is not same");
    break;
  }
}

template double solver::solver<double>::get_residual(vector<double> &x);
template float solver::solver<float>::get_residual(vector<float> &x);
} // namespace monolish
