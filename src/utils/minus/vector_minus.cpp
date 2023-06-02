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
  if (get_device_mem_stat() == true) {
    ans.send();
  }

  internal::vmul(size(), begin(), -1, ans.begin(), get_device_mem_stat());
  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator-();
template vector<float> vector<float>::operator-();

} // namespace monolish
