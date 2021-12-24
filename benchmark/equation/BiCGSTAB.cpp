#include "../benchmark_utils.hpp"

template <typename MAT_A, typename T, typename SOLVER, typename PRECOND>
bool benchmark(const size_t size, const size_t iter) {

  monolish::matrix::COO<T> COO =
      monolish::util::laplacian_matrix_2D_5p<T>(size, size);
  MAT_A A(COO);
  monolish::vector<T> x(A.get_row(), 123.0);
  monolish::vector<T> b(A.get_row(), 1.0);

  // std::cout << A.get_row() << "," << A.get_nnz() << std::endl;

  monolish::util::send(A, x, b);

  SOLVER solver;

  solver.set_miniter(ITERARIVE_SOLVER_ITER);
  solver.set_maxiter(ITERARIVE_SOLVER_ITER);
  solver.set_tol(-100);
  // ITERARIVE_SOLVER_solver.set_print_rhistory(true);

  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    bool flag = monolish::util::solver_check(solver.solve(A, x, b));
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();
  x.device_free();
  b.device_free();

  double time = sec / iter;
  std::cout << solver.solver_name() << "(" << precond.solver_name() << ","
            << A.type() << ")\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << size << "\t" << std::flush;
  std::cout << ITERARIVE_SOLVER_ITER << "\t" << std::flush;
  std::cout << time << "\t" << std::flush;
  std::cout << time / ITERARIVE_SOLVER_ITER << "\t" << std::endl;

  return true;
}

int main(int argc, char **argv) {

  if (argc <= 1) {
    std::cout << "error $1: format of A (only CRS now)" << std::endl;
    return 1;
  }

  if ((strcmp(argv[1], "CRS") != 0)) {
    return 1;
  }

  std::cout << "func\tprec\tsize\titer\ttime[sec]\ttime/iter[sec]" << std::endl;

  size_t iter = ITERARIVE_SOLVER_BENCH_ITER;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
       size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
       size ITERARIVE_SOLVER_NN_BENCH_ITER) {
    benchmark<monolish::matrix::CRS<float>, float,
              monolish::equation::BiCGSTAB<monolish::matrix::CRS<float>, float>,
              monolish::equation::none<monolish::matrix::CRS<float>, float>>(
        size, iter);
  }

  for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
       size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
       size ITERARIVE_SOLVER_NN_BENCH_ITER) {
    benchmark<
        monolish::matrix::CRS<double>, double,
        monolish::equation::BiCGSTAB<monolish::matrix::CRS<double>, double>,
        monolish::equation::none<monolish::matrix::CRS<double>, double>>(size,
                                                                         iter);
  }

  for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
       size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
       size ITERARIVE_SOLVER_NN_BENCH_ITER) {
    benchmark<monolish::matrix::CRS<float>, float,
              monolish::equation::BiCGSTAB<monolish::matrix::CRS<float>, float>,
              monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
        size, iter);
  }

  for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
       size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
       size ITERARIVE_SOLVER_NN_BENCH_ITER) {
    benchmark<
        monolish::matrix::CRS<double>, double,
        monolish::equation::BiCGSTAB<monolish::matrix::CRS<double>, double>,
        monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
        size, iter);
  }

  for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
       size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
       size ITERARIVE_SOLVER_NN_BENCH_ITER) {
    benchmark<monolish::matrix::CRS<float>, float,
              monolish::equation::BiCGSTAB<monolish::matrix::CRS<float>, float>,
              monolish::equation::SOR<monolish::matrix::CRS<float>, float>>(
        size, iter);
  }

  for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
       size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
       size ITERARIVE_SOLVER_NN_BENCH_ITER) {
    benchmark<
        monolish::matrix::CRS<double>, double,
        monolish::equation::BiCGSTAB<monolish::matrix::CRS<double>, double>,
        monolish::equation::SOR<monolish::matrix::CRS<double>, double>>(size,
                                                                        iter);
  }

  if (monolish::util::build_with_gpu() == true) {
    for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
         size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
         size ITERARIVE_SOLVER_NN_BENCH_ITER) {
      benchmark<
          monolish::matrix::CRS<float>, float,
          monolish::equation::BiCGSTAB<monolish::matrix::CRS<float>, float>,
          monolish::equation::IC<monolish::matrix::CRS<float>, float>>(size,
                                                                       iter);
    }

    for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
         size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
         size ITERARIVE_SOLVER_NN_BENCH_ITER) {
      benchmark<
          monolish::matrix::CRS<double>, double,
          monolish::equation::BiCGSTAB<monolish::matrix::CRS<double>, double>,
          monolish::equation::IC<monolish::matrix::CRS<double>, double>>(size,
                                                                         iter);
    }
  }

  if (monolish::util::build_with_gpu() == true) {
    for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
         size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
         size ITERARIVE_SOLVER_NN_BENCH_ITER) {
      benchmark<
          monolish::matrix::CRS<float>, float,
          monolish::equation::BiCGSTAB<monolish::matrix::CRS<float>, float>,
          monolish::equation::ILU<monolish::matrix::CRS<float>, float>>(size,
                                                                        iter);
    }

    for (size_t size = ITERARIVE_SOLVER_NN_BENCH_MIN;
         size <= ITERARIVE_SOLVER_NN_BENCH_MAX;
         size ITERARIVE_SOLVER_NN_BENCH_ITER) {
      benchmark<
          monolish::matrix::CRS<double>, double,
          monolish::equation::BiCGSTAB<monolish::matrix::CRS<double>, double>,
          monolish::equation::ILU<monolish::matrix::CRS<double>, double>>(size,
                                                                          iter);
    }
  }

  return 0;
}
