#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> void ans_exp(monolish::matrix::Dense<T> &A) {

  for (size_t i = 0; i < A.get_nnz(); i++) {
    A.data()[i] = std::exp(A.data()[i]);
  }
}

template <typename MAT, typename T>
bool test_send_mexp(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  ans_exp(AA);

  A.send();
  monolish::vml::exp(A, A);
  A.recv();
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.data(), AA.data(),
                      AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_mexp(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  ans_exp(AA);

  monolish::vml::exp(A, A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.data(), AA.data(),
                      AA.get_nnz(), tol);
}