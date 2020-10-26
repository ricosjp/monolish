#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// copy
template <typename T> Dense<T> Dense<T>::copy() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat()) {
    nonfree_recv();
  } // gpu copy

  Dense<T> tmp;
  std::copy(val.data(), val.data() + nnz, tmp.val.begin());
  tmp.rowN = get_row();
  tmp.colN = get_col();
  tmp.nnz = get_nnz();
  if (get_device_mem_stat()) {
    tmp.send();
  } // gpu copy

  logger.util_out();
  return tmp;
}

template Dense<double> Dense<double>::copy();
template Dense<float> Dense<float>::copy();

// copy monolish Dense
template <typename T> void Dense<T>::operator=(const Dense<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(mat.get_nnz());

  rowN = mat.get_row();
  colN = mat.get_col();
  nnz = mat.get_nnz();

  // gpu copy and recv
  if (mat.get_device_mem_stat()) {
#if MONOLISH_USE_GPU
    send();
    size_t NNZ = nnz;
    T *vald = val.data();
    const T *Mvald = mat.val.data();

#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < NNZ; i++) {
      vald[i] = Mvald[i];
    }

    nonfree_recv();
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    std::copy(mat.val.data(), mat.val.data() + nnz, val.begin());
  }

  logger.util_out();
}

template void Dense<double>::operator=(const Dense<double> &mat);
template void Dense<float>::operator=(const Dense<float> &mat);

} // namespace matrix
} // namespace monolish
