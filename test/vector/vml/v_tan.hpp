#include "../../test_utils.hpp"

template <typename T> void ans_vtan(monolish::vector<T> &ans) {
  for (size_t i = 0; i < ans.size(); i++) {
    ans[i] = std::tan(ans[i]);
  }
}

template <typename T> bool test_send_vtan(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());

  monolish::vector<T> ans_tmp(ans);
  ans_vtan(ans_tmp);

  monolish::util::send(ans);
  monolish::vml::tan(ans, ans);
  ans.recv();

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), ans.size(), tol);
}

template <typename T> bool test_vtan(const size_t size, double tol) {

  monolish::vector<T> ans(size, 0.1, 1.0, test_random_engine());

  monolish::vector<T> ans_tmp(ans);
  ans_vtan(ans_tmp);

  monolish::vml::tan(ans, ans);

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), ans.size(), tol);
}
