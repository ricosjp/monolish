#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T>
void ans_axpy(double alpha, monolish::vector<T> &mx, monolish::vector<T> &my) {
  if (mx.size() != my.size()) {
    std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    my[i] = alpha * mx[i] + my[i];
  }
}

template <typename T> bool test_send_axpy(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  monolish::vector<T> ansy = y;
  ans_axpy(alpha, x, ansy);

  monolish::util::send(x, y);
  monolish::blas::axpy(alpha, x, y);
  y.recv();

  if (ans_check<T>(y.data(), ansy.data(), y.size(), tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}

template <typename T> bool test_axpy(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  monolish::vector<T> ansy = y;
  ans_axpy(alpha, x, ansy);

  monolish::blas::axpy(alpha, x, y);

  if (ans_check<T>(y.data(), ansy.data(), y.size(), tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}
