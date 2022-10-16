#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
/////////////////////////////////////////////////
// vec - scalar
/////////////////////////////////////////////////
template <typename T> vector<T> vector<T>::operator-() {

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(size());
  if (gpu_status == true) {
    ans.send();
  }

  internal::vmul(size(), vad, -1, ans.data(), gpu_status);
  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator-();
template vector<float> vector<float>::operator-();

} // namespace monolish
