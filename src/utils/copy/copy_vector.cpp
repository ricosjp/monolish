#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T> void vector<T>::operator=(const std::vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(vec.size());
  internal::vcopy(vec.size(), vec.data(), begin(), false);

  logger.util_out();
}

template void vector<double>::operator=(const std::vector<double> &vec);
template void vector<float>::operator=(const std::vector<float> &vec);

template <typename T> void vector<T>::operator=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));
  val_create_flag = true;

  // gpu copy and recv
  internal::vcopy(vec.size(), vec.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}

template void vector<double>::operator=(const vector<double> &vec);
template void vector<float>::operator=(const vector<float> &vec);

template <typename T>
void vector<T>::operator=(const view1D<vector<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));
  val_create_flag = true;

  // gpu copy and recv
  internal::vcopy(vec.size(), vec.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}
template void
vector<double>::operator=(const view1D<vector<double>, double> &vec);
template void vector<float>::operator=(const view1D<vector<float>, float> &vec);

template <typename T>
void vector<T>::operator=(const view1D<matrix::Dense<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));
  val_create_flag = true;

  // gpu copy and recv
  internal::vcopy(vec.size(), vec.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}
template void
vector<double>::operator=(const view1D<matrix::Dense<double>, double> &vec);
template void
vector<float>::operator=(const view1D<matrix::Dense<float>, float> &vec);

template <typename T>
void vector<T>::operator=(const view1D<tensor::tensor_Dense<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));
  val_create_flag = true;

  // gpu copy and recv
  internal::vcopy(vec.size(), vec.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}
template void vector<double>::operator=(
    const view1D<tensor::tensor_Dense<double>, double> &vec);
template void
vector<float>::operator=(const view1D<tensor::tensor_Dense<float>, float> &vec);

} // namespace monolish
