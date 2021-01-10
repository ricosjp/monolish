#include "../../test_utils.hpp"

template <typename T> void ans_vsinh(monolish::vector<T> &ans) {
  for (size_t i = 0; i < ans.size(); i++) {
    ans[i] = std::sinh(ans[i]);
  }
}

template <typename T> bool test_send_vsinh(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 10.0);

  monolish::vector<T> ans_tmp(ans);
  ans_vsinh(ans_tmp);

  monolish::util::send(ans);
  monolish::vml::sinh(ans, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), ans.size(), tol);
}

template <typename T> bool test_vsinh(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 10.0);

  monolish::vector<T> ans_tmp(ans);
  ans_vsinh(ans_tmp);

  monolish::vml::sinh(ans, ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), ans.size(), tol);
}
