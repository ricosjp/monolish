#include "../../test_utils.hpp"

template <typename T> T ans_vmax(monolish::vector<T> &ans) {
  return *(std::max_element(ans.begin(), ans.end()));
}

template <typename T> bool test_send_vmax(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());

  T ans = ans_vmax(x);

  monolish::util::send(x);
  T result = monolish::vml::max(x);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename T> bool test_vmax(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());

  T ans = ans_vmax(x);

  T result = monolish::vml::max(x);

  return ans_check<T>(__func__, result, ans, tol);
}
