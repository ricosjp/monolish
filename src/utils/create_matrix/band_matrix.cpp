#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
matrix::COO<T> util::band_matrix(const int M, const int N, const int W,
                                 const T diag_val, const T val) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (N <= W) {
    throw std::runtime_error("error band width <= matrix size");
  }

  matrix::COO<T> mat(M, N);

  int ww = W;

  for (auto i = decltype(M){0}; i < M; i++) {
    auto start = (i < ww ? 0 : i - ww);
    auto end = (N <= (i + ww + 1) ? N : i + ww + 1);
    for (auto j = start; j < end; j++) {
      if (i == j && diag_val != 0) {
        mat.insert(i, j, diag_val);
      }
      else if (val != 0){
        mat.insert(i, j, val);
      }
    }
  }

  logger.util_out();

  return mat;
}
template matrix::COO<double> util::band_matrix(const int M, const int N,
                                               const int W,
                                               const double diag_val,
                                               const double val);
template matrix::COO<float> util::band_matrix(const int M, const int N,
                                              const int W, const float diag_val,
                                              const float val);

template <typename T>
matrix::COO<T> util::asym_band_matrix(const int M, const int N, const int W,
                                 const T diag_val, const T Uval, const T Lval) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (N <= W) {
    throw std::runtime_error("error band width <= matrix size");
  }

  matrix::COO<T> mat(M, N);

  int ww = W;

  for (auto i = decltype(M){0}; i < M; i++) {
    auto start = (i < ww ? 0 : i - ww);
    auto end = (N <= (i + ww + 1) ? N : i + ww + 1);
    for (auto j = start; j < end; j++) {
      if (i == j && diag_val != 0){
        mat.insert(i, j, diag_val);
      }
      else if(i < j && Uval != 0){ // Upper
        mat.insert(i, j, Uval);
      }
      else if(i > j && Lval != 0){ // Lower
        mat.insert(i, j, Lval);
      }
    }
  }

  logger.util_out();

  return mat;
}
template matrix::COO<double> util::asym_band_matrix(const int M, const int N,
                                               const int W,
                                               const double diag_val,
                                               const double Uval,
                                               const double Lval);
template matrix::COO<float> util::asym_band_matrix(const int M, const int N,
                                              const int W, const float diag_val,
                                              const float Uval,
                                              const float Lval);
} // namespace monolish
