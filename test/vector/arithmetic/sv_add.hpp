#include "../../test_utils.hpp"

template <typename T>
void ans_svadd(monolish::vector<T> &mx, T value, monolish::vector<T> &ans) {
  if (mx.size() != ans.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] = mx[i] + value;
  }
}

template <typename T> bool test_send_svadd(const size_t size, double tol) {

  T value = 123.0;
  monolish::vector<T> x(size, 0.1, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp = ans.copy();
  ans_svadd(x, value, ans_tmp);

  monolish::util::send(x, ans);
  monolish::blas::add(x, value, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}

template <typename T> bool test_svadd(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  T value = 123.0;
  monolish::vector<T> x(size, 0.1, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp = ans.copy();
  ans_svadd(x, value, ans_tmp);

  ans += x + value;

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}
