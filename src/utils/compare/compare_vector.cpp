#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
bool vector<T>::equal(const vector<T> &vec, bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (val.size() != vec.size()) {
    return false;
  }
  if (get_device_mem_stat() != vec.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(val.size(), val.data(), vec.data(), true))) {
      return false;
    }
  } else if (get_device_mem_stat() == false ||
             compare_cpu_and_device == false) {
    if (!(internal::vequal(val.size(), val.data(), vec.data(), false))) {
      return false;
    }
  }

  logger.util_out();
  return true;
}
template bool vector<double>::equal(const vector<double> &vec,
                                    bool compare_cpu_and_device) const;
template bool vector<float>::equal(const vector<float> &vec,
                                   bool compare_cpu_and_device) const;

template <typename T> bool vector<T>::operator==(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(vec, false);

  logger.util_out();
  return ans;
}
template bool vector<double>::operator==(const vector<double> &vec);
template bool vector<float>::operator==(const vector<float> &vec);

template <typename T> bool vector<T>::operator!=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(vec, false);

  logger.util_out();
  return !(ans);
}
template bool vector<double>::operator!=(const vector<double> &vec);
template bool vector<float>::operator!=(const vector<float> &vec);

} // namespace monolish
