
#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
tensor::tensor_COO<T> util::random_structure_tensor(const size_t M, const size_t N, const size_t L, 
                                             const size_t nnzrow, const T val) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (L <= nnzrow) {
    throw std::runtime_error("error nnzrow <= tensor size");
  }

  tensor::tensor_COO<T> tens({M, N, L});

  std::random_device seed;
  std::default_random_engine rng(seed());

  for (auto i = decltype(M){0}; i < M; i++) {
    for(auto j = decltype(N){0}; j < N; j++) {
      std::uniform_int_distribution<> dist_pos(0, L - 1);
      for (auto k = decltype(nnzrow){0}; k < nnzrow; k++) {
        size_t c = dist_pos(rng);
        if (tens.at({i, j, c}) != 0)
          k--;
        tens.insert({i, j, c}, val);
      }
    }
  }

  tens.sort(true);

  logger.util_out();

  return tens;
}
template tensor::tensor_COO<double> util::random_structure_tensor(const size_t M,
                                                           const size_t N,
                                                           const size_t L,
                                                           const size_t nnzrow,
                                                           const double val);
template tensor::tensor_COO<float> util::random_structure_tensor(const size_t M,
                                                          const size_t N,
                                                          const size_t L,
                                                          const size_t nnzrow,
                                                          const float val);

template <typename T>
tensor::tensor_COO<T> util::random_structure_tensor(const size_t K, const size_t M, const size_t N, const size_t L, 
                                             const size_t nnzrow, const T val) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (L <= nnzrow) {
    throw std::runtime_error("error nnzrow <= tensor size");
  }

  tensor::tensor_COO<T> tens({K, M, N, L});

  std::random_device seed;
  std::default_random_engine rng(seed());

  for (auto t = decltype(K){0}; t<K; t++){
    for (auto i = decltype(M){0}; i < M; i++) {
      for(auto j = decltype(N){0}; j < N; j++) {
        std::uniform_int_distribution<> dist_pos(0, L - 1);
        for (auto k = decltype(nnzrow){0}; k < nnzrow; k++) {
          size_t c = dist_pos(rng);
          if (tens.at({t, i, j, c}) != 0)
            k--;
          tens.insert({t, i, j, c}, val);
        }
      }
    }
  }

  tens.sort(true);

  logger.util_out();

  return tens;
}
template tensor::tensor_COO<double> util::random_structure_tensor(const size_t K,
                                                           const size_t M,
                                                           const size_t N,
                                                           const size_t L,
                                                           const size_t nnzrow,
                                                           const double val);
template tensor::tensor_COO<float> util::random_structure_tensor(const size_t K,
                                                          const size_t M,
                                                          const size_t N,
                                                          const size_t L,
                                                          const size_t nnzrow,
                                                          const float val);
} // namespace monolish
