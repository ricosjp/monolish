#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// diag
template <typename T> void LinearOperator<T>::diag(vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vecd = vec.data();

  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = std::min(get_row(), get_col());

  assert(Len == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vector<T> vec_tmp(N, 0);
      vector<T> ans(M);
      vec_tmp[i] = 1;
      util::send(vec_tmp);
      ans = matvec(vec_tmp);
      util::recv(ans);
      const T *ansd = ans.data();
      vecd[i] = ansd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vector<T> vec_tmp(N, 0);
      vector<T> ans(M);
      vec_tmp[i] = 1;
      ans = matvec(vec_tmp);
      vecd[i] = ans[i];
    }
  }

  logger.func_out();
}

template void
monolish::matrix::LinearOperator<double>::diag(vector<double> &vec) const;
template void
monolish::matrix::LinearOperator<float>::diag(vector<float> &vec) const;

template <typename T>
void LinearOperator<T>::diag(view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vecd = vec.data();

  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = std::min(get_row(), get_col());

  assert(Len == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vector<T> vec_tmp(N, 0);
      vector<T> ans(M);
      vec_tmp[i] = 1;
      util::send(vec_tmp);
      ans = matvec(vec_tmp);
      util::recv(ans);
      const T *ansd = ans.data();
      vecd[i] = ansd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vector<T> vec_tmp(N, 0);
      vector<T> ans(M);
      vec_tmp[i] = 1;
      ans = matvec(vec_tmp);
      vecd[i] = ans[i];
    }
  }

  logger.func_out();
}

template void monolish::matrix::LinearOperator<double>::diag(
    view1D<vector<double>, double> &vec) const;
template void monolish::matrix::LinearOperator<float>::diag(
    view1D<vector<float>, float> &vec) const;

template <typename T>
void LinearOperator<T>::diag(view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vecd = vec.data();

  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = std::min(get_row(), get_col());

  assert(Len == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vector<T> vec_tmp(N, 0);
      vector<T> ans(M);
      vec_tmp[i] = 1;
      util::send(vec_tmp);
      ans = matvec(vec_tmp);
      util::recv(ans);
      const T *ansd = ans.data();
      vecd[i] = ansd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vector<T> vec_tmp(N, 0);
      vector<T> ans(M);
      vec_tmp[i] = 1;
      ans = matvec(vec_tmp);
      vecd[i] = ans[i];
    }
  }

  logger.func_out();
}

template void monolish::matrix::LinearOperator<double>::diag(
    view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::LinearOperator<float>::diag(
    view1D<matrix::Dense<float>, float> &vec) const;

} // namespace matrix
} // namespace monolish
