#include "../benchmark_utils.hpp"

#define FUNC "CG"
#define DENSE_PERF                                                             \
  2.0 / 3.0 *                                                                  \
      ((double)size / 1000 * (double)size / 1000 * (double)size / 1000) / time

template <typename MAT_A, typename T>
bool benchmark(const size_t size, const size_t iter) {

  monolish::matrix::COO<T> COO =
      monolish::util::laplacian_matrix_2D_5p<T>(size, size);
  MAT_A A(COO);
  monolish::vector<T> x(A.get_row(), 123.0);
  monolish::vector<T> b(A.get_row(), 1.0);

  // std::cout << A.get_row() << "," << A.get_nnz() << std::endl;

  monolish::util::send(A, x, b);

  monolish::equation::CG<MAT_A, T> CG_solver;

  CG_solver.set_miniter(CG_ITER);
  CG_solver.set_maxiter(CG_ITER);
  CG_solver.set_tol(-100);
  // CG_solver.set_print_rhistory(true);

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    CG_solver.solve(A, x, b);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();
  x.device_free();
  b.device_free();

  double time = sec / iter;
  std::cout << FUNC << "(" << A.type() << ")\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << size << "\t" << std::flush;
  std::cout << CG_ITER << "\t" << std::flush;
  std::cout << time << "\t" << std::flush;
  std::cout << time / CG_ITER << "\t" << std::endl;

  return true;
}

int main(int argc, char **argv) {

  if (argc <= 1) {
    std::cout << "error $1: format of A (only Dense now)" << std::endl;
    return 1;
  }

  if ((strcmp(argv[1], "CRS") != 0)) {
    return 1;
  }

  std::cout << "func\ttprec\tsize\titer\ttime[sec]\ttime/iter[sec]"
            << std::endl;

  size_t iter = CG_BENCH_ITER;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  // CRS
  for (size_t size = CG_NN_BENCH_MIN; size <= CG_NN_BENCH_MAX;
       size CG_NN_BENCH_ITER) {
    benchmark<monolish::matrix::CRS<float>, float>(size, iter);
  }

  for (size_t size = CG_NN_BENCH_MIN; size <= CG_NN_BENCH_MAX;
       size CG_NN_BENCH_ITER) {
    benchmark<monolish::matrix::CRS<double>, double>(size, iter);
  }

  return 0;
}
