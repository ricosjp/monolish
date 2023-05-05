#include "../../test_utils.hpp"

template <typename T>
void ans_valo(monolish::vector<T> &ans, const T alpha, const T beta) {
  for (size_t i = 0; i < ans.size(); i++) {
    if (ans[i] > 0) {
      ans[i] = alpha * ans[i];
    } else {
      ans[i] = beta * ans[i];
    }
  }
}

template <typename T> bool test_send_valo(const size_t size, double tol) {

  monolish::vector<T> ans(size, -1.0, 1.0, test_random_engine());
  T alpha = 1.5;
  T beta = 0.5;
  monolish::vector<T> ans_tmp(ans);

  ans_valo(ans_tmp, alpha, beta);

  monolish::util::send(ans);
  monolish::vml::alo(ans, alpha, beta, ans);
  monolish::util::recv(ans);

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), ans.size(), tol);
}

template <typename T> bool test_valo(const size_t size, double tol) {

  monolish::vector<T> ans(size, -1.0, 1.0, test_random_engine());
  T alpha = 1.5;
  T beta = 0.5;

  monolish::vector<T> ans_tmp(ans);
  ans_valo(ans_tmp, alpha, beta);

  monolish::vml::alo(ans, alpha, beta, ans);

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), ans.size(), tol);
}
