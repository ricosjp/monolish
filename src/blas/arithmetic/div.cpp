#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
/////////////////////////////////////////////////
// vec - vec
/////////////////////////////////////////////////
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

/////////////////////////////////////////////////
// vec - scalar
/////////////////////////////////////////////////
template <typename T>
  void blas::div(const vector<T>& a, const T alpha, vector<T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (a.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (a.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(y.size(), a.data(), alpha, y.data(), y.gpu_status);

  logger.func_out();
}

/////////////////////////////////////////////////
// vec - scalar
/////////////////////////////////////////////////
template <typename T> vector<T> vector<T>::operator/(const T value) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(val.size());
  if (gpu_status == true) {
    ans.send();
  }

  internal::vdiv(val.size(), val.data(), value, ans.data(), gpu_status);

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator/(const double value);
template vector<float> vector<float>::operator/(const float value);

template <typename T> void vector<T>::operator/=(const T value) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vdiv(val.size(), val.data(), value, val.data(), gpu_status);

  logger.func_out();
}
template void vector<double>::operator/=(const double value);
template void vector<float>::operator/=(const float value);

/////////////////////////////////////////////////
// vec - vec
/////////////////////////////////////////////////

template <typename T> vector<T> vector<T>::operator/(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (val.size() != vec.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (gpu_status != vec.gpu_status) {
    throw std::runtime_error("error gpu_status is not same");
  }

  vector<T> ans(vec.size());
  if (gpu_status == true) {
    ans.send();
  }

  internal::vdiv(val.size(), val.data(), vec.data(), ans.data(), gpu_status);

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator/(const vector<double> &vec);
template vector<float> vector<float>::operator/(const vector<float> &vec);

template <typename T> void vector<T>::operator/=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (val.size() != vec.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (gpu_status != vec.gpu_status) {
    throw std::runtime_error("error gpu_status is not same");
  }

  internal::vdiv(val.size(), val.data(), vec.data(), val.data(), gpu_status);

  logger.func_out();
}
template void vector<double>::operator/=(const vector<double> &vec);
template void vector<float>::operator/=(const vector<float> &vec);
} // namespace monolish
