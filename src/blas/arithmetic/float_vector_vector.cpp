#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
  void blas::add(const vector<T>& a, const vector<T>& b, vector<T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != b.size() || a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != b.get_device_mem_stat() || a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vadd(y.size(), a.data(), b.data, y.data(), y.gpu_status);

  logger.func_out();
}

template <typename T>
  void blas::sub(const vector<T>& a, const vector<T>& b, vector<T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != b.size() || a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != b.get_device_mem_stat() || a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsub(y.size(), a.data(), b.data, y.data(), y.gpu_status);

  logger.func_out();
}

template <typename T>
  void blas::mul(const vector<T>& a, const vector<T>& b, vector<T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != b.size() || a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != b.get_device_mem_stat() || a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmul(y.size(), a.data(), b.data, y.data(), y.gpu_status);

  logger.func_out();
}

template <typename T>
  void blas::div(const vector<T>& a, const vector<T>& b, vector<T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != b.size() || a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != b.get_device_mem_stat() || a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(y.size(), a.data(), b.data, y.data(), y.gpu_status);

  logger.func_out();
}

} // namespace monolish
