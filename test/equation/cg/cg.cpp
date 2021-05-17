#include "../../test_utils.hpp"
#include "../include/monolish_blas.hpp"
#include "../include/monolish_equation.hpp"

template <typename MATRIX, typename T, typename PRECOND>
bool test(const char *file, const int check_ans, const T tol) {

  monolish::matrix::COO<T> COO(file);
  MATRIX A(COO);

  monolish::vector<T> ans(A.get_row(), 1.0);
  monolish::vector<T> b(A.get_row(), 0.0);

  // initial x is rand(0~1)
  monolish::vector<T> x(A.get_row(), 0, 1.0);

  monolish::util::send(A, x, b, ans);

  // create answer
  monolish::blas::matvec(A, ans, b);

  monolish::equation::CG<MATRIX, T> solver;

  solver.set_tol(tol);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(10000);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);
  // solver.set_rhistory_filename("./rhistroy.txt");

  if (monolish::util::solver_check(solver.solve(A, x, b))) {
    return false;
  }

  // std::cout << monolish::util::get_residual_l2(A,x,b) << std::endl;

  ans.recv();
  x.recv();

  if (check_ans == 1) {
    if (ans_check<T>(x.data(), ans.data(), x.size(), tol) == false) {
      x.print_all();
      return false;
    };
  }
  return true;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
    return 1;
  }

  char *file = argv[1];
  int check_ans = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  std::cout << "CRS, none" << std::endl;

  if (test<monolish::matrix::CRS<double>, double,
           monolish::equation::none<monolish::matrix::CRS<double>, double>>(
          file, check_ans, 1.0e-8) == false) {
    return 1;
  }
  if (test<monolish::matrix::CRS<float>, float,
           monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          file, check_ans, 1.0e-4) == false) {
    return 1;
  }

  std::cout << "CRS, jacobi" << std::endl;

  if (test<monolish::matrix::CRS<double>, double,
           monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
          file, check_ans, 1.0e-8) == false) {
    return 1;
  }
  if (test<monolish::matrix::CRS<float>, float,
           monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
          file, check_ans, 1.0e-4) == false) {
    return 1;
  }

  std::cout << "Dense, none" << std::endl;

  if (test<monolish::matrix::Dense<double>, double,
           monolish::equation::none<monolish::matrix::Dense<double>, double>>(
          file, check_ans, 1.0e-8) == false) {
    return 1;
  }
  if (test<monolish::matrix::Dense<float>, float,
           monolish::equation::none<monolish::matrix::Dense<float>, float>>(
          file, check_ans, 1.0e-4) == false) {
    return 1;
  }

  std::cout << "Dense, jacobi" << std::endl;

  if (test<monolish::matrix::Dense<double>, double,
           monolish::equation::Jacobi<monolish::matrix::Dense<double>, double>>(
          file, check_ans, 1.0e-8) == false) {
    return 1;
  }
  if (test<monolish::matrix::Dense<float>, float,
           monolish::equation::Jacobi<monolish::matrix::Dense<float>, float>>(
          file, check_ans, 1.0e-4) == false) {
    return 1;
  }

  if (monolish::util::build_with_gpu() == false) {
    std::cout << "LinearOperator, none" << std::endl;

    if (test<monolish::matrix::LinearOperator<double>, double,
             monolish::equation::none<monolish::matrix::LinearOperator<double>,
                                      double>>(file, check_ans, 1.0e-8) ==
        false) {
      return 1;
    }
    if (test<monolish::matrix::LinearOperator<float>, float,
             monolish::equation::none<monolish::matrix::LinearOperator<float>,
                                      float>>(file, check_ans, 1.0e-4) ==
        false) {
      return 1;
    }
  }

  if (monolish::util::build_with_gpu() == false) {
    std::cout << "LinearOperator, jacobi" << std::endl;

    if (test<monolish::matrix::LinearOperator<double>, double,
             monolish::equation::Jacobi<
                 monolish::matrix::LinearOperator<double>, double>>(
            file, check_ans, 1.0e-8) == false) {
      return 1;
    }
    if (test<monolish::matrix::LinearOperator<float>, float,
             monolish::equation::Jacobi<monolish::matrix::LinearOperator<float>,
                                        float>>(file, check_ans, 1.0e-4) ==
        false) {
      return 1;
    }
  }

  return 0;
}
