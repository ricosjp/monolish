#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

// copy std vector
template <typename T> void vector<T>::operator=(const std::vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(vec.size());
  std::copy(vec.begin(), vec.end(), val.begin());

  logger.util_out();
}

template void vector<double>::operator=(const std::vector<double> &vec);
template void vector<float>::operator=(const std::vector<float> &vec);

// copy monolish vector
template <typename T> void vector<T>::operator=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // gpu copy and recv
  if (vec.get_device_mem_stat()) {

    if (get_device_mem_stat() == false) {
      throw std::runtime_error(
          "Error, No GPU memory allocated for the return value (operator=)");
    }
    if (vec.size() != size()) {
      throw std::runtime_error("error vector size is not same");
    }

#if MONOLISH_USE_GPU
    size_t size = vec.size();
    T *vald = val.data();
    const T *vecd = vec.data();
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
      vald[i] = vecd[i];
    }
#endif
  } else {
    val.resize(vec.size());
    std::copy(vec.val.begin(), vec.val.end(), val.begin());
  }

  logger.util_out();
}

template void vector<double>::operator=(const vector<double> &vec);
template void vector<float>::operator=(const vector<float> &vec);

// copy constructor
template <typename T> vector<T>::vector(const monolish::vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(vec.size());

  // gpu copy and recv
  if (vec.get_device_mem_stat()) {
    send();

#if MONOLISH_USE_GPU
    size_t size = vec.size();
    T *vald = val.data();
    const T *vecd = vec.data();
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
      vald[i] = vecd[i];
    }

    nonfree_recv();
#endif
  } else {
    std::copy(vec.val.begin(), vec.val.end(), val.begin());
  }

  logger.util_out();
}
template vector<double>::vector(const vector<double> &vec);
template vector<float>::vector(const vector<float> &vec);
} // namespace monolish
