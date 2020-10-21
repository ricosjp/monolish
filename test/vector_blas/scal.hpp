#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> void ans_scal(double alpha, monolish::vector<T> &mx) {

  for (size_t i = 0; i < mx.size(); i++) {
    mx[i] = alpha * mx[i];
  }
}

template <typename T> bool test_send_scal(const size_t size, double tol) {

  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);

  monolish::vector<T> ansx = x;
  ans_scal(alpha, ansx);

  x.send();
  monolish::blas::scal(alpha, x);
  x.recv();

  if (ans_check<T>(x.data(), ansx.data(), x.size(), tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}

template <typename T> bool test_scal(const size_t size, double tol) {

  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);

  monolish::vector<T> ansx = x;
  ans_scal(alpha, ansx);

  monolish::blas::scal(alpha, x);

  if (ans_check<T>(x.data(), ansx.data(), x.size(), tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}
