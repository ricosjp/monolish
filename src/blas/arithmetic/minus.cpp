#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

namespace monolish {
/////////////////////////////////////////////////
// vec - scalar
/////////////////////////////////////////////////
template <typename T> vector<T> vector<T>::operator-() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(val.size());

  T *vald = val.data();
  T *ansd = ans.data();
  size_t size = val.size();

#if MONOLISH_USE_GPU
  ans.send();
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < size; i++) {
    ansd[i] = -vald[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    ansd[i] = -vald[i];
  }
#endif

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator-();
template vector<float> vector<float>::operator-();

} // namespace monolish
