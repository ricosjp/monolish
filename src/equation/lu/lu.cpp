#include "../../../include/monolish_equation.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <>
int equation::LU<matrix::CRS<double>, double>::solve(matrix::CRS<double> &A, vector<double> &x,
                                vector<double> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  (void)(&A);
  (void)(&x);
  (void)(&b);
  logger.func_out();
  throw std::runtime_error("error solver.lib is not 1");
}

} // namespace monolish
