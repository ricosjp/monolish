#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
//  ///////////////////
void blas::vecadd(const vector<float> &a, const vector<float> &b,
                  vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  blas::add(a, b, y);

  logger.func_out();
}

// vecsub ///////////////////
void blas::vecsub(const vector<float> &a, const vector<float> &b,
                  vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  blas::sub(a, b, y);

  logger.func_out();
}

} // namespace monolish
