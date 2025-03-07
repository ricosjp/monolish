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

template <typename T> void tensor_CRS<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
    const T *vald = begin();

    std::vector<int> pos_2(shape.size() - 2);
    int nz = 0;
    for (size_t d = 0; d < row_ptrs.size(); ++d) {
      size_t d_copy = d;
      for (int k = (int)shape.size() - 3; k >= 0; --k) {
        pos_2[k] = d_copy % shape[k];
        d_copy /= shape[k];
      }
      const auto *indexd = col_inds[d].data();
      const auto *ptrd = row_ptrs[d].data();
#pragma omp target
      for (size_t i = 0; i < row_ptrs[d].size(); i++) {
        for (size_t j = ptrd[i]; j < ptrd[i + 1]; j++) {
          for (size_t k = 0; k < pos_2.size(); ++k) {
            printf("%lu ", pos_2[k] + 1);
          }
          printf("%lu %d %f\n", i + 1, indexd[j] + 1, vald[nz]);
          nz++;
        }
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    std::vector<int> pos_2(shape.size() - 2);
    int nz = 0;
    for (size_t d = 0; d < row_ptrs.size(); ++d) {
      size_t d_copy = d;
      for (int k = (int)shape.size() - 3; k >= 0; --k) {
        pos_2[k] = d_copy % shape[k];
        d_copy /= shape[k];
      }
      for (size_t i = 0; i < row_ptrs[d].size(); i++) {
        for (auto j = row_ptrs[d][i]; j < row_ptrs[d][i + 1]; j++) {
          for (size_t k = 0; k < pos_2.size(); ++k) {
            std::cout << pos_2[k] + 1 << " ";
          }
          std::cout << i + 1 << " " << col_inds[d][j] + 1 << " " << begin()[nz]
                    << std::endl;
          nz++;
        }
      }
    }
  }

  logger.util_out();
}
template void tensor_CRS<double>::print_all(bool force_cpu) const;
template void tensor_CRS<float>::print_all(bool force_cpu) const;

} // namespace tensor
} // namespace monolish
