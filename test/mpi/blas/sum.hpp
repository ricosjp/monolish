#include "../../test_utils.hpp"
#include "monolish_mpi.hpp"

template <typename T> T ans_sum(monolish::vector<T> &mx) {
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i];
  }

  monolish::mpi::comm &comm = monolish::mpi::comm::get_instance();
  ans = comm.Allreduce(ans);

  return ans;
}

template <typename T> bool test_send_sum(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0);

  auto ans = ans_sum(x);
  monolish::util::send(x);
  auto result = monolish::blas::sum(x);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename T> bool test_sum(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0);

  auto result = monolish::blas::sum(x);
  auto ans = ans_sum(x);

  return ans_check<T>(__func__, result, ans, tol);
}
