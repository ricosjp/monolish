#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "subvec_op"
#define DENSE_PERF 2 * M *N / time / 1.0e+9
#define CRS_PERF 2 * M *nnzrow / time / 1.0e+9

template <typename T>
void get_ans(monolish::vector<T> &vec) {

  for (int i = 0; i < vec.size(); i++) {
      vec[i] = vec[i] + 3;
      vec[i] = vec[i] - 3;
      vec[i] = vec[i] * 3;
      vec[i] = vec[i] / 3;
  }
}

template <typename T>
bool test(const size_t M, const size_t N, double tol) {
    int nnzrow=3;
    monolish::matrix::Dense<T> A(M, N, 1.0, 2.0); // M*N matrix

    monolish::vector<T> vec;
    monolish::vector<T> ansvec;

    //////////////////////////
    //
    vec = monolish::vector<T>(A.get_col(), 0.0, 1.0);
    ansvec = monolish::vector<T>(A.get_col(), 0.0, 1.0);

    A.row(A.get_col()-1, ansvec);
    get_ans(ansvec);

    A.row_add(1,3.0);
    A.row_mul(1,3.0);
    A.row_div(1,3.0);
    A.row_sub(1,3.0);
    A.row(A.get_col()-1, vec);

    if (ans_check<T>(ansvec.data(), vec.data(), vec.size(), tol) == false) {
        std::cout << "===" << std::endl;
        ansvec.print_all();
        std::cout << "===" << std::endl;
        vec.print_all();
      return false;
    };

    //////////////////////////

    vec = monolish::vector<T>(A.get_row(), 0.0, 1.0);
    ansvec = monolish::vector<T>(A.get_row(), 0.0, 1.0);

    A.col(A.get_col()-1, ansvec);
    get_ans(ansvec);

    A.col_add(1,3.0);
    A.col_mul(1,3.0);
    A.col_div(1,3.0);
    A.col_sub(1,3.0);
    A.col(A.get_col()-1, vec);

    if (ans_check<T>(ansvec.data(), vec.data(), vec.size(), tol) == false) {
        std::cout << "===" << std::endl;
        ansvec.print_all();
        std::cout << "===" << std::endl;
        vec.print_all();
      return false;
    };

    //////////////////////////

    size_t n = A.get_row() > A.get_col() ? A.get_row() : A.get_col();
    vec = monolish::vector<T>(n, 0.0, 1.0);
    ansvec = monolish::vector<T>(n, 0.0, 1.0);

    A.diag(ansvec);
    get_ans(ansvec);

    A.diag_add(3.0);
    A.diag_mul(3.0);
    A.diag_div(3.0);
    A.diag_sub(3.0);
    A.diag(vec);

    if (ans_check<T>(ansvec.data(), vec.data(), vec.size(), tol) == false) {
        std::cout << "===" << std::endl;
        ansvec.print_all();
        std::cout << "===" << std::endl;
        vec.print_all();
      return false;
    };
    std::cout << "Pass in " << get_type<T>() << " precision" << std::endl;

    return true;
}

int main(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "error $1: precision (double or float) $2: row, $3: col"
            << std::endl;
        return 1;
    }

    const size_t M = atoi(argv[2]);
    const size_t N = atoi(argv[3]);

    // monolish::util::set_log_level(3);
    // monolish::util::set_log_filename("./monolish_test_log.txt");

    if (strcmp(argv[1], "double") == 0) {
        if (test<double>(M, N, 1.0e-8) == false) {
            return 1;
        }
    }

    if (strcmp(argv[1], "float") == 0) {
        if (test<float>(M, N, 1.0e-5) == false) {
            return 1;
        }
    }
    return 0;
}
