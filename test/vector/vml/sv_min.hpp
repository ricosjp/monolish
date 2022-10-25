#include "../../test_utils.hpp"

template <typename T>
void ans_svmin(monolish::vector<T> &mx, const T alpha,
               monolish::vector<T> &ans) {
  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] = std::min(mx[i], alpha);
  }
}

template <typename T> bool test_send_svmin(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());
  T alpha = 0.5;
  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans_tmp(ans);

  ans_svmin(x, alpha, ans_tmp);

  monolish::util::send(x, ans);
  monolish::vml::min(x, alpha, ans);
  monolish::util::recv(ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}

template <typename T> bool test_svmin(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());
  T alpha = 0.5;
  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans_tmp(ans);

  ans_svmin(x, alpha, ans_tmp);

  monolish::vml::min(x, alpha, ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}
