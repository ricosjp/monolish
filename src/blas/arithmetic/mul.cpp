#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
/////////////////////////////////////////////////
// vec - scalar
/////////////////////////////////////////////////
template <typename T> vector<T> vector<T>::operator*(const T value) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(val.size());
  if (gpu_status == true) {
    ans.send();
  }

  internal::vmul(val.size(), val.data(), value, ans.data(), gpu_status);

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator*(const double value);
template vector<float> vector<float>::operator*(const float value);

template <typename T> void vector<T>::operator*=(const T value) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(val.size(), val.data(), value, val.data(), gpu_status);

  logger.func_out();
}
template void vector<double>::operator*=(const double value);
template void vector<float>::operator*=(const float value);

/////////////////////////////////////////////////
// vec - vec
/////////////////////////////////////////////////

template <typename T> vector<T> vector<T>::operator*(const vector<T> &vec) {
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

  internal::vmul(val.size(), val.data(), vec.data(), ans.data(), gpu_status);

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator*(const vector<double> &vec);
template vector<float> vector<float>::operator*(const vector<float> &vec);

template <typename T> void vector<T>::operator*=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (val.size() != vec.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (gpu_status != vec.gpu_status) {
    throw std::runtime_error("error gpu_status is not same");
  }

  internal::vmul(val.size(), val.data(), vec.data(), val.data(), gpu_status);

  logger.func_out();
}
template void vector<double>::operator*=(const vector<double> &vec);
template void vector<float>::operator*=(const vector<float> &vec);
} // namespace monolish
