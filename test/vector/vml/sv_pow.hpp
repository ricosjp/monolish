#include "../../test_utils.hpp"

template <typename T>
void ans_svpow(monolish::vector<T> &mx, T value, monolish::vector<T> &ans) {
  if (mx.size() != ans.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] = std::pow(mx[i], value);
  }
}

template <typename T> bool test_send_svpow(const size_t size, double tol) {

  T value = 123.0;
  monolish::vector<T> x(size, 1.0, 2.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp(ans);
  ans_svpow(x, value, ans_tmp);

  monolish::util::send(x, ans);
  monolish::vml::pow(x, value, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), x.size(), tol);
}

template <typename T> bool test_svpow(const size_t size, double tol) {
  std::cout << __func__ << "(" << get_type<T>() << ")" << std::flush;

  T value = 123.0;
  monolish::vector<T> x(size, 1.0, 2.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp(ans);
  ans_svpow(x, value, ans_tmp);

  monolish::vml::pow(x, value, ans);

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), x.size(), tol);
}
