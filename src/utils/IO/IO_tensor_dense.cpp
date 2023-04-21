#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {
namespace tensor {

template <typename T> void tensor_Dense<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
    const T *vald = data();
#pragma omp target
    for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
      auto ind = get_index(i);
      for (auto j = decltype(ind.size()){0}; j < ind.size(); j++) {
        printf("%lu ", ind[j]);
      }
      printf("%f\n", vald[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
      auto ind = get_index(i);
      for (auto j = decltype(ind.size()){0}; j < ind.size(); j++) {
        std::cout << ind[j] + 1 << " ";
      }
      std::cout << data()[i] << std::endl;
    }
  }

  logger.util_out();
}
template void tensor_Dense<double>::print_all(bool force_cpu) const;
template void tensor_Dense<float>::print_all(bool force_cpu) const;

} // namespace tensor
} // namespace monolish
