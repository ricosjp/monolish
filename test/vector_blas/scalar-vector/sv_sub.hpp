#include "../../test_utils.hpp"

template <typename T>
void ans_svsub(monolish::vector<T> &mx, T value, monolish::vector<T> &ans) {
  if (mx.size() != ans.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    ans[i] -= mx[i] - value;
  }
}

template <typename T>
bool test_send_svsub(const size_t size, double tol){

  T value = 123.0;
  monolish::vector<T> x(size, 0.1, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp = ans.copy();
  ans_svsub(x, value, ans_tmp);

  monolish::util::send(x, ans);
  ans -= x - value;
  ans.recv();

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}

template <typename T>
bool test_svsub(const size_t size, double tol){

  T value = 123.0;
  monolish::vector<T> x(size, 0.1, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::vector<T> ans_tmp = ans.copy();
  ans_svsub(x, value, ans_tmp);

  ans -= x - value;

  return ans_check<T>(__func__, ans.data(), ans_tmp.data(), x.size(), tol);
}
