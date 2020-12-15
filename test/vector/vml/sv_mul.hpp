#include "../../test_utils.hpp"

template <typename T>
void ans_svmul(monolish::vector<T> &mx, T value, monolish::vector<T> &ans) {
  if (mx.size() != ans.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] = mx[i] * value;
  }
}

template <typename T> bool test_send_svmul(const size_t size, double tol) {

  T value = 123.0;
  monolish::vector<T> x(size, 0.1, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp = ans.copy();
  ans_svmul(x, value, ans_tmp);

  monolish::util::send(x, ans);
  monolish::vml::mul(x, value, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}

template <typename T> bool test_svmul(const size_t size, double tol) {

  T value = 123.0;
  monolish::vector<T> x(size, 0.1, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp = ans.copy();
  ans_svmul(x, value, ans_tmp);

  monolish::vml::mul(x, value, ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}
