#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T> bool vector<T>::operator==(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (val.size() != vec.size()) {
    return false;
  }

  if (get_device_mem_stat() != vec.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    bool ret = internal::vequal(val.size(), val.data(), vec.data(), true);
    if (ret == false) {
      return false;
    }
  }
  bool ret = internal::vequal(val.size(), val.data(), vec.data(), false);

  logger.util_out();
  return ret;
}
template bool vector<double>::operator==(const vector<double> &vec);
template bool vector<float>::operator==(const vector<float> &vec);

template <typename T> bool vector<T>::operator!=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (val.size() != vec.size()) {
    return true;
  }

  if (get_device_mem_stat() != vec.get_device_mem_stat()) {
    return true;
  }

  if (get_device_mem_stat() == true) {
    bool ret = internal::vequal(val.size(), val.data(), vec.data(), true);
    if (ret == false) {
      return true;
    }
  }
  bool ret = internal::vequal(val.size(), val.data(), vec.data(), false);

  logger.util_out();
  return !ret;
}
template bool vector<double>::operator!=(const vector<double> &vec);
template bool vector<float>::operator!=(const vector<float> &vec);

}
