#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

void blas::add(const vector<double> &a, const double alpha, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vadd(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void blas::sub(const vector<double> &a, const double alpha, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsub(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void blas::mul(const vector<double> &a, const double alpha, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmul(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

void blas::div(const vector<double> &a, const double alpha, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
