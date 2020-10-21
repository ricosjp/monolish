#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> T ans_nrm2(monolish::vector<T> &mx) {
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i] * mx[i];
  }
  ans = sqrt(ans);

  return ans;
}

template <typename T> bool test_send_nrm2(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  monolish::vector<T> x(size, 0.0, 1.0);

  auto ans = ans_nrm2(x);
  monolish::util::send(x);
  auto result = monolish::blas::nrm2(x);

  if (ans_check<T>(result, ans, tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}

template <typename T> bool test_nrm2(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  monolish::vector<T> x(size, 0.0, 1.0);

  auto result = monolish::blas::nrm2(x);
  auto ans = ans_nrm2(x);

  if (ans_check<T>(result, ans, tol) == false) {
    std::cout << ": fail" << std::endl;
    return false;
  }

  std::cout << ": pass" << std::endl;

  return true;
}
