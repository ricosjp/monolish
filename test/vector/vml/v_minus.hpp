#include "../../test_utils.hpp"

template <typename T>
void ans_minus(monolish::vector<T> &mx, monolish::vector<T> &ans) {

  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] = -mx[i];
  }
}

template <typename T> bool test_send_minus(const size_t size, double tol) {

  // create random vector x rand(0.1~1.0)
  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp(ans);
  ans_minus(x, ans_tmp);

  monolish::util::send(x, ans);
  ans = -x;
  ans.recv();

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), x.size(), tol);
}

template <typename T> bool test_minus(const size_t size, double tol) {

  // create random vector x rand(0.1~1.0)
  monolish::vector<T> x(size, 0.1, 1.0, test_random_engine());
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp(ans);
  ans_minus(x, ans_tmp);

  ans = -x;

  return ans_check<T>(__func__, ans.begin(), ans_tmp.begin(), x.size(), tol);
}
