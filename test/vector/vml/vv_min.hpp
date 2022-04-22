#include "../../test_utils.hpp"

template <typename T>
void ans_vvmin(monolish::vector<T> &mx, monolish::vector<T> &my,
               monolish::vector<T> &ans) {
  if (mx.size() != my.size()) {
    throw std::runtime_error("x.size != y.size");
  }
  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] = std::min(mx[i], my[i]);
  }
}

template <typename T> bool test_send_vvmin(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans_tmp(ans);

  ans_vvmin(x, y, ans_tmp);

  monolish::util::send(x, y, ans);
  monolish::vml::min(x, y, ans);
  monolish::util::recv(ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}

template <typename T> bool test_vvmin(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans_tmp(ans);

  ans_vvmin(x, y, ans_tmp);

  monolish::vml::min(x, y, ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}
