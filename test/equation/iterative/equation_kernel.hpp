#include "../../test_utils.hpp"
#include "../include/monolish_blas.hpp"
#include "../include/monolish_equation.hpp"

template <typename MATRIX, typename T, typename SOLVER, typename PRECOND>
bool test(const char *file, const int check_ans, const T tol) {

  monolish::matrix::COO<T> COO(file);
  MATRIX A(COO);

  monolish::vector<T> ans(A.get_row(), 1.0);
  monolish::vector<T> b(A.get_row(), 0.0);

  // initial x is rand(0~1)
  monolish::vector<T> x(A.get_row(), 0, 0.1);

  monolish::util::send(A, x, b, ans);

  // create answer
  monolish::blas::matvec(A, ans, b);

  SOLVER solver;

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
    if (ans_check<T>(x.data(), ans.data(), x.size(), tol * 10) == false) {
      x.print_all();
      return false;
    };
  }
  return true;
}
