#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {
namespace matrix {

template <typename T> void CRS<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_GPU
#pragma omp target
    for (size_t i = 0; i < get_row(); i++) {
      for (size_t j = (size_t)row_ptr[i]; j < (size_t)row_ptr[i + 1]; j++) {
        printf("%d %d %f\n", i + 1, col_ind[j] + 1, val[j]);
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (size_t i = 0; i < get_row(); i++) {
      for (size_t j = (size_t)row_ptr[i]; j < (size_t)row_ptr[i + 1]; j++) {
        std::cout << i + 1 << " " << col_ind[j] + 1 << " " << val[j]
                  << std::endl;
      }
    }
  }

  logger.util_out();
}
template void CRS<double>::print_all(bool force_cpu) const;
template void CRS<float>::print_all(bool force_cpu) const;

} // namespace matrix
} // namespace monolish
