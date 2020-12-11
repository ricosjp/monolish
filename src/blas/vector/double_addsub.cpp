#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
//  ///////////////////
void blas::vecadd(const vector<double> &a, const vector<double> &b, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  blas::add(a, b, y);

  logger.func_out();
}

// vecsub ///////////////////
void blas::vecsub(const vector<double> &a, const vector<double> &b, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  blas::sub(a, b, y);

  logger.func_out();
}

} // namespace monolish
