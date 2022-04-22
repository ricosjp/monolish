#include "../../test_utils.hpp"

template <typename T> T ans_nrm2(monolish::vector<T> &mx) {
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i] * mx[i];
  }
  ans = sqrt(ans);

  return ans;
}

template <typename T> bool test_send_nrm2(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0, test_random_engine());

  auto ans = ans_nrm2(x);
  monolish::util::send(x);
  auto result = monolish::blas::nrm2(x);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename T> bool test_nrm2(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0, test_random_engine());

  auto result = monolish::blas::nrm2(x);
  auto ans = ans_nrm2(x);

  return ans_check<T>(__func__, result, ans, tol);
}
