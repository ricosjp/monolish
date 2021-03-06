#include "../../test_utils.hpp"

template <typename T>
T ans_dot(monolish::vector<T> &mx, monolish::vector<T> &my) {
  if (mx.size() != my.size()) {
    throw std::runtime_error("x.size != y.size");
  }
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i] * my[i];
  }

  return ans;
}

template <typename T> bool test_send_dot(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.0, 1.0, test_random_engine());

  T ans = ans_dot(x, y);

  monolish::util::send(x, y);
  T result = monolish::blas::dot(x, y);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename T> bool test_dot(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.0, 1.0, test_random_engine());

  T ans = ans_dot(x, y);

  monolish::util::send(x, y);
  T result = monolish::blas::dot(x, y);

  return ans_check<T>(__func__, result, ans, tol);
}
