#include "../../test_utils.hpp"

template <typename T>
void ans_valo(monolish::vector<T> &mx, const T alpha, const T beta,
                monolish::vector<T> &ans) {
  for(size_t i = 0; i < mx.size(); i++) {
    if (mx[i] > 0) {
      ans[i] = alpha * mx[i];
    } else {
      ans[i] = beta * mx[i];
    }
  }
}

template <typename T>
bool test_send_valo(const size_t size, double tol) {

  monolish::vector<T> x(size, -1.0, 1.0, test_random_engine());
  T alpha = 1.5;
  T beta = 0.5;
  monolish::vector<T> ans(size, -1.0, 1.0, test_random_engine());
  monolish::vector<T> ans_tmp(ans);

  ans_valo(x, alpha, beta, ans_tmp);

  monolish::util::send(x, ans);
  monolish::vml::alo(x, alpha, beta, ans);
  monolish::util::recv(ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}

template <typename T> bool test_valo(const size_t size, double tol) {

  monolish::vector<T> x(size, -1.0, 1.0, test_random_engine());
  T alpha = 1.5;
  T beta = 0.5;
  monolish::vector<T> ans(size, -1.0, 1.0, test_random_engine());
  monolish::vector<T> ans_tmp(ans);

  ans_valo(x, alpha, beta, ans_tmp);

  monolish::vml::alo(x, alpha, beta, ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}
