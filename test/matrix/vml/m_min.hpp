#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> T ans_min(monolish::matrix::Dense<T> &A) {
  return *(std::min_element(A.val.begin(), A.val.end()));
}

template <typename MAT, typename T>
bool test_send_mmin(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, -1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  T ans = ans_min(AA);

  A.send();
  T result = monolish::vml::min(A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename MAT, typename T>
bool test_mmin(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, -1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  T ans = ans_min(AA);

  T result = monolish::vml::min(A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, result, ans, tol);
}
