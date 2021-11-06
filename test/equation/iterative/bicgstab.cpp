#include "./equation_kernel.hpp"
#define SOLVER BiCGSTAB
#define SOLVER_NAME "BiCGSTAB"
#define D_TOL 1.0e-8
#define S_TOL 1.0e-4
#define PRECOND 1

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
    return 1;
  }

  char *file = argv[1];
  int check_ans = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  ///////////////////////////////////////////////////////////
  // SOLVER
  ///////////////////////////////////////////////////////////
  std::cout << "CRS, " SOLVER_NAME << ", none" << std::endl;

  if (test<monolish::matrix::CRS<double>, double,
           monolish::equation::SOLVER<monolish::matrix::CRS<double>, double>,
           monolish::equation::none<monolish::matrix::CRS<double>, double>>(
          file, check_ans, D_TOL) == false) {
    return 1;
  }
  if (test<monolish::matrix::CRS<float>, float,
           monolish::equation::SOLVER<monolish::matrix::CRS<float>, float>,
           monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          file, check_ans, S_TOL) == false) {
    return 1;
  }

#if PRECOND == 1
  std::cout << "CRS, " SOLVER_NAME << ", Jacobi" << std::endl;

  if (test<monolish::matrix::CRS<double>, double,
           monolish::equation::SOLVER<monolish::matrix::CRS<double>, double>,
           monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
          file, check_ans, D_TOL) == false) {
    return 1;
  }
  if (test<monolish::matrix::CRS<float>, float,
           monolish::equation::SOLVER<monolish::matrix::CRS<float>, float>,
           monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
          file, check_ans, S_TOL) == false) {
    return 1;
  }

  std::cout << "CRS, " SOLVER_NAME << ", SOR" << std::endl;

  if (test<monolish::matrix::CRS<double>, double,
           monolish::equation::SOLVER<monolish::matrix::CRS<double>, double>,
           monolish::equation::SOR<monolish::matrix::CRS<double>, double>>(
          file, check_ans, D_TOL) == false) {
    return 1;
  }
  if (test<monolish::matrix::CRS<float>, float,
           monolish::equation::SOLVER<monolish::matrix::CRS<float>, float>,
           monolish::equation::SOR<monolish::matrix::CRS<float>, float>>(
          file, check_ans, S_TOL) == false) {
    return 1;
  }

  if (monolish::util::build_with_gpu() == true) {
    std::cout << "CRS, " SOLVER_NAME << ", ILU" << std::endl;

    if (test<monolish::matrix::CRS<double>, double,
             monolish::equation::SOLVER<monolish::matrix::CRS<double>, double>,
             monolish::equation::ILU<monolish::matrix::CRS<double>, double>>(
            file, check_ans, D_TOL) == false) {
      return 1;
    }
    if (test<monolish::matrix::CRS<float>, float,
             monolish::equation::SOLVER<monolish::matrix::CRS<float>, float>,
             monolish::equation::ILU<monolish::matrix::CRS<float>, float>>(
            file, check_ans, S_TOL) == false) {
      return 1;
    }
  }
#endif

  std::cout << "Dense, " SOLVER_NAME << ", none" << std::endl;

  if (test<monolish::matrix::Dense<double>, double,
           monolish::equation::SOLVER<monolish::matrix::Dense<double>, double>,
           monolish::equation::none<monolish::matrix::Dense<double>, double>>(
          file, check_ans, D_TOL) == false) {
    return 1;
  }
  if (test<monolish::matrix::Dense<float>, float,
           monolish::equation::SOLVER<monolish::matrix::Dense<float>, float>,
           monolish::equation::none<monolish::matrix::Dense<float>, float>>(
          file, check_ans, S_TOL) == false) {
    return 1;
  }

#if PRECOND == 1
  std::cout << "Dense, " SOLVER_NAME << ", Jacobi" << std::endl;

  if (test<monolish::matrix::Dense<double>, double,
           monolish::equation::SOLVER<monolish::matrix::Dense<double>, double>,
           monolish::equation::Jacobi<monolish::matrix::Dense<double>, double>>(
          file, check_ans, D_TOL) == false) {
    return 1;
  }
  if (test<monolish::matrix::Dense<float>, float,
           monolish::equation::SOLVER<monolish::matrix::Dense<float>, float>,
           monolish::equation::Jacobi<monolish::matrix::Dense<float>, float>>(
          file, check_ans, S_TOL) == false) {
    return 1;
  }

  std::cout << "Dense, " SOLVER_NAME << ", SOR" << std::endl;

  if (test<monolish::matrix::Dense<double>, double,
           monolish::equation::SOLVER<monolish::matrix::Dense<double>, double>,
           monolish::equation::SOR<monolish::matrix::Dense<double>, double>>(
          file, check_ans, D_TOL) == false) {
    return 1;
  }
  if (test<monolish::matrix::Dense<float>, float,
           monolish::equation::SOLVER<monolish::matrix::Dense<float>, float>,
           monolish::equation::SOR<monolish::matrix::Dense<float>, float>>(
          file, check_ans, S_TOL) == false) {
    return 1;
  }
#endif

  if (monolish::util::build_with_gpu() == false) {
    std::cout << "LinearOperator, " SOLVER_NAME << ", none" << std::endl;

    if (test<monolish::matrix::LinearOperator<double>, double,
             monolish::equation::SOLVER<
                 monolish::matrix::LinearOperator<double>, double>,
             monolish::equation::none<monolish::matrix::LinearOperator<double>,
                                      double>>(file, check_ans, D_TOL) ==
        false) {
      return 1;
    }
    if (test<monolish::matrix::LinearOperator<float>, float,
             monolish::equation::SOLVER<monolish::matrix::LinearOperator<float>,
                                        float>,
             monolish::equation::none<monolish::matrix::LinearOperator<float>,
                                      float>>(file, check_ans, S_TOL) ==
        false) {
      return 1;
    }
  }

#if PRECOND == 1
  if (monolish::util::build_with_gpu() == false) {
    std::cout << "LinearOperator, " SOLVER_NAME << ", Jacobi" << std::endl;

    if (test<monolish::matrix::LinearOperator<double>, double,
             monolish::equation::SOLVER<
                 monolish::matrix::LinearOperator<double>, double>,
             monolish::equation::Jacobi<
                 monolish::matrix::LinearOperator<double>, double>>(
            file, check_ans, D_TOL) == false) {
      return 1;
    }
    if (test<monolish::matrix::LinearOperator<float>, float,
             monolish::equation::SOLVER<monolish::matrix::LinearOperator<float>,
                                        float>,
             monolish::equation::Jacobi<monolish::matrix::LinearOperator<float>,
                                        float>>(file, check_ans, S_TOL) ==
        false) {
      return 1;
    }
    //     std::cout << "LinearOperator, " SOLVER_NAME << ", SOR" << std::endl;
    //
    //     if (test<monolish::matrix::LinearOperator<double>, double,
    //              monolish::equation::SOLVER<
    //                  monolish::matrix::LinearOperator<double>, double>,
    //              monolish::equation::SOR<
    //                  monolish::matrix::LinearOperator<double>, double>>(
    //             file, check_ans, D_TOL) == false) {
    //       return 1;
    //     }
    //     if (test<monolish::matrix::LinearOperator<float>, float,
    //              monolish::equation::SOLVER<monolish::matrix::LinearOperator<float>,
    //                                         float>,
    //              monolish::equation::SOR<monolish::matrix::LinearOperator<float>,
    //                                         float>>(file, check_ans, S_TOL)
    //                                         ==
    //         false) {
    //       return 1;
    //     }
  }
#endif

  return 0;
}
