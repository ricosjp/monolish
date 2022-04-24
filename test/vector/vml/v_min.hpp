#include "../../test_utils.hpp"

template <typename T> T ans_vmin(monolish::vector<T> &ans) {
  return *(std::min_element(ans.begin(), ans.end()));
}

template <typename T> bool test_send_vmin(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());

  T ans = ans_vmin(x);

  monolish::util::send(x);
  T result = monolish::vml::min(x);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename T> bool test_vmin(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());

  T ans = ans_vmin(x);

  T result = monolish::vml::min(x);

  return ans_check<T>(__func__, result, ans, tol);
}
