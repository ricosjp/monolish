#include "../test_utils.hpp"

template <typename MAT, typename T>
bool test_send_transpose(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  if (A.type() != "COO")
    monolish::util::send(A);

  A.transpose();

  if (A.type() != "COO")
    A.recv();

  if (A.get_row() != N || A.get_col() != M) {
    std::cout << "transpose error, transA.row = " << A.get_row()
              << ", transA.col = " << A.get_col() << std::endl;
    return false;
  }

  if (A.type() != "COO")
    monolish::util::send(A);
  A.transpose();
  if (A.type() != "COO")
    A.recv();

  monolish::matrix::COO<T> ansA(A);
  if (ans_check<T>(__func__, A.type(), seedA.val.data(), ansA.val.data(),
                   ansA.get_nnz(), tol) == false) {
    return false;
  };

  return true;
}

template <typename MAT, typename T>
bool test_transpose(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  A.transpose();

  if (A.get_row() != N || A.get_col() != M) {
    std::cout << "transpose error, transA.row = " << A.get_row()
              << ", transA.col = " << A.get_col() << std::endl;
    return false;
  }

  A.transpose();

  monolish::matrix::COO<T> ansA(A);
  if (ans_check<T>(__func__, A.type(), seedA.val.data(), ansA.val.data(),
                   ansA.get_nnz(), tol) == false) {
    return false;
  };

  return true;
}

template <typename MAT, typename T>
bool test_transpose_elements(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  MAT B(N, M); // N*M matrix
  B.transpose(A);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::string is = std::to_string(i);
      std::string js = std::to_string(j);
      std::string s = "A(";
      s += is;
      s += ",";
      s += js;
      s += ") == A^T(";
      s += js;
      s += ",";
      s += is;
      s += ")";
      if (std::abs(A.at(i, j)) > tol &&
          ans_check<T>(s, A.at(i, j), B.at(j, i), tol) == false) {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "$1: row, $2: col" << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  std::cout << "M=" << M << ", N=" << N << std::endl;

  if (test_send_transpose<monolish::matrix::Dense<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::Dense<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::COO<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::COO<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::Dense<double>, double>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::Dense<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::COO<double>, double>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::COO<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::Dense<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::Dense<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::COO<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::COO<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  return 0;
}
