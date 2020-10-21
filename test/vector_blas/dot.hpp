#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T>
T ans_dot(monolish::vector<T> &mx, monolish::vector<T> &my) {
  if (mx.size() != my.size()) {
    std::runtime_error("x.size != y.size");
  }
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i] * my[i];
  }

  return ans;
}

template <typename T> bool test_send_dot(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  // create random vector x rand(0~1)
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  T ans = ans_dot(x, y);

  monolish::util::send(x, y);
  T result = monolish::blas::dot(x, y);

  if (ans_check<T>(result, ans, tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}

template <typename T> bool test_dot(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  // create random vector x rand(0~1)
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  T ans = ans_dot(x, y);

  monolish::util::send(x, y);
  T result = monolish::blas::dot(x, y);

  if (ans_check<T>(result, ans, tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}
