#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// scalar-tensor times //
template <typename T>
void times_core(const T alpha, const tensor::tensor_CRS<T> &A,
                tensor::tensor_CRS<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.begin(), alpha, C.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// vector-tensor_row times all //
template <typename T, typename VEC>
void times_row_core(const tensor::tensor_CRS<T> &A, const VEC &x,
                    tensor::tensor_CRS<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  std::vector<size_t> Ashape = A.get_shape();
  assert(Ashape[Ashape.size() - 1] == x.size());

  const auto *Ad = A.begin();
  auto *Cd = C.begin();

  auto nsum = 0;

  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    const auto *rowd = A.row_ptrs[d].data();
    const auto *cold = A.col_inds[d].data();
    const int n = (int)A.row_ptrs[d].size() - 1;

    const auto *xd = x.begin();

    if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
      for (auto i = decltype(n){0}; i < n; i++) {
        for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
          Cd[j + nsum] = Ad[j + nsum] * xd[cold[j]];
        }
      }
#else
      throw std::runtime_error(
          "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
    } else {
#pragma omp parallel for
      for (auto i = decltype(n){0}; i < n; i++) {
        for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
          Cd[j + nsum] = Ad[j + nsum] * xd[cold[j]];
        }
      }
    }
    nsum += A.col_inds[d].size();
  }

  logger.func_out();
}

} // namespace
} // namespace monolish
