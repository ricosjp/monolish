#include "../../test_utils.hpp"

template <typename T> void ans_vsqrt(monolish::vector<T> &ans) {
  for (size_t i = 0; i < ans.size(); i++) {
    ans[i] = std::sqrt(ans[i]);
  }
}

template <typename T> bool test_send_vsqrt(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());

  monolish::vector<T> ans_tmp(ans);
  ans_vsqrt(ans_tmp);

  monolish::util::send(ans);
  monolish::vml::sqrt(ans, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), ans.size(), tol);
}

template <typename T> bool test_vsqrt(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());

  monolish::vector<T> ans_tmp(ans);
  ans_vsqrt(ans_tmp);

  monolish::vml::sqrt(ans, ans);

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), ans.size(), tol);
}
