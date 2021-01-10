#include "../../test_utils.hpp"

template <typename T> void ans_vatanh(monolish::vector<T> &ans) {
  for (size_t i = 0; i < ans.size(); i++) {
    ans[i] = std::atanh(ans[i]);
  }
}

template <typename T> bool test_send_vatanh(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 10.0);

  monolish::vector<T> ans_tmp(ans);
  ans_vatanh(ans_tmp);

  monolish::util::send(ans);
  monolish::vml::atanh(ans, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), ans.size(), tol);
}

template <typename T> bool test_vatanh(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 10.0);

  monolish::vector<T> ans_tmp(ans);
  ans_vatanh(ans_tmp);

  monolish::vml::atanh(ans, ans);

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), ans.size(), tol);
}
