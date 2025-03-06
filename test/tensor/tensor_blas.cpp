#include "blas/adds_col.hpp"
#include "blas/adds_col_line.hpp"
#include "blas/adds_row.hpp"
#include "blas/adds_row_line.hpp"
#include "blas/mattens.hpp"
#include "blas/scalar_adds.hpp"
#include "blas/scalar_times.hpp"
#include "blas/tensadd.hpp"
#include "blas/tensmat.hpp"
#include "blas/tensmul.hpp"
#include "blas/tenssub.hpp"
#include "blas/tensvec.hpp"
#include "blas/times_col.hpp"
#include "blas/times_col_line.hpp"
#include "blas/times_row.hpp"
#include "blas/times_row_line.hpp"
#include "blas/tscal.hpp"
#include "blas/tt_copy.hpp"

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cout << "error!, $1:M, $2:N, $3:K, $4:L, $5:J" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t M = atoi(argv[1]);
  size_t N = atoi(argv[2]);
  size_t K = atoi(argv[3]);
  size_t L = atoi(argv[4]);
  size_t J = atoi(argv[5]);
  std::cout << "M=" << M << ", N=" << N << ", K=" << K << ", L=" << L
            << ", J=" << J << std::endl;

  // tensadd tensor_Dense//
  if (test_send_tensadd<monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tensadd<monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tensadd<monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tensadd<monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_tensadd_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tensadd_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tensadd_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tensadd_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tenssub tensor_Dense//
  if (test_send_tenssub<monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tenssub<monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tenssub<monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tenssub<monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_tenssub_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tenssub_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tenssub_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tenssub_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tensvec tensor_Dense//
  if (test_send_tensvec<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_tensvec<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-3) == false) {
    return 1;
  }
  if (test_tensvec<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-6) == false) {
    return 1;
  }
  if (test_tensvec<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-3) == false) {
    return 1;
  }
  if (test_send_tensvec_view<double>(M, N, K, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_tensvec_view<float>(M, N, K, 1.0e-3) == false) {
    return 1;
  }
  if (test_tensvec_view<double>(M, N, K, 1.0e-6) == false) {
    return 1;
  }
  if (test_tensvec_view<float>(M, N, K, 1.0e-3) == false) {
    return 1;
  }

  // tensmat tensor_Dense//
  if (test_send_tensmat<monolish::tensor::tensor_Dense<double>,
                        monolish::matrix::Dense<double>,
                        monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tensmat<monolish::tensor::tensor_Dense<float>,
                        monolish::matrix::Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tensmat<monolish::tensor::tensor_Dense<double>,
                   monolish::matrix::Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tensmat<monolish::tensor::tensor_Dense<float>,
                   monolish::matrix::Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_tensmat_view<double>(M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tensmat_view<float>(M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tensmat_view<double>(M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tensmat_view<float>(M, N, K, L, 1.0e-4) == false) {
    return 1;
  }

  // mattens Dense tensor_Dense//
  if (test_send_mattens<

          monolish::matrix::Dense<double>,
          monolish::tensor::tensor_Dense<double>,
          monolish::tensor::tensor_Dense<double>, double>(M, N, K, L, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mattens<monolish::matrix::Dense<float>,
                        monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_mattens<monolish::matrix::Dense<double>,
                   monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_mattens<monolish::matrix::Dense<float>,
                   monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_mattens_view_dense<double>(M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_mattens_view_dense<float>(M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_mattens_view_dense<double>(M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_mattens_view_dense<float>(M, N, K, L, 1.0e-4) == false) {
    return 1;
  }

  // mattens CRS tensor_Dense//
  if (test_send_mattens<

          monolish::matrix::CRS<double>, monolish::tensor::tensor_Dense<double>,
          monolish::tensor::tensor_Dense<double>, double>(M, N, K, L, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mattens<monolish::matrix::CRS<float>,
                        monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_mattens<monolish::matrix::CRS<double>,
                   monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_mattens<monolish::matrix::CRS<float>,
                   monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_mattens_view_crs<double>(M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_mattens_view_crs<float>(M, N, K, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_mattens_view_crs<double>(M, N, K, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_mattens_view_crs<float>(M, N, K, L, 1.0e-4) == false) {
    return 1;
  }

  // tensmul tensor_Dense//
  if (test_send_tensmul<monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, L, J, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tensmul<monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, J, 1.0e-4) == false) {
    return 1;
  }
  if (test_tensmul<monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, L, J, 1.0e-8) == false) {
    return 1;
  }
  if (test_tensmul<monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, L, J, 1.0e-4) == false) {
    return 1;
  }

  // tt_copy tensor_Dense//
  if (test_send_tt_copy<monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_copy<monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_copy<monolish::tensor::tensor_Dense<double>,
                   monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_copy<monolish::tensor::tensor_Dense<float>,
                   monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_copy_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_copy_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_copy_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_copy_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }

  // tt_copy tensor_CRS//
  if (test_send_tt_copy<monolish::tensor::tensor_CRS<double>,
                        monolish::tensor::tensor_CRS<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_copy<monolish::tensor::tensor_CRS<float>,
                        monolish::tensor::tensor_CRS<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_copy<monolish::tensor::tensor_CRS<double>,
                   monolish::tensor::tensor_CRS<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_copy<monolish::tensor::tensor_CRS<float>,
                   monolish::tensor::tensor_CRS<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }

  // tscal tensor_Dense//
  if (test_send_tscal<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tscal<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tscal<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tscal<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tscal_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tscal_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_tscal_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_tscal_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }

  // scalar_times tensor_Dense//
  if (test_send_scalar_times<monolish::tensor::tensor_Dense<double>,
                             monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_scalar_times<monolish::tensor::tensor_Dense<float>,
                             monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_scalar_times<monolish::tensor::tensor_Dense<double>,
                        monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_scalar_times<monolish::tensor::tensor_Dense<float>,
                        monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_scalar_times_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_scalar_times_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_scalar_times_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_scalar_times_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }

  // scalar_adds tensor_Dense//
  if (test_send_scalar_adds<monolish::tensor::tensor_Dense<double>,
                            monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_scalar_adds<monolish::tensor::tensor_Dense<float>,
                            monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_scalar_adds<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_scalar_adds<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_scalar_adds_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_scalar_adds_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }
  if (test_scalar_adds_view<double>(M, N, L, 1.0e-8) == false) {
    return 1;
  }
  if (test_scalar_adds_view<float>(M, N, L, 1.0e-4) == false) {
    return 1;
  }

  // vector_times row tensor_Dense//
  if (test_send_times_row<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_row<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_row<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_row<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_times_row_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_row_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_row_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_row_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_times col tensor_Dense//
  if (test_send_times_col<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_col<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_col<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_col<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_times_col_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_col_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_col_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_col_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_times row_line tensor_Dense//
  if (test_send_times_row_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_row_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_row_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_row_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_times_row_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_row_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_row_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_row_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_times col_line tensor_Dense//
  if (test_send_times_col_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_col_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_col_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_col_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_times_col_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_times_col_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_times_col_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_times_col_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_adds row tensor_Dense//
  if (test_send_adds_row<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_row<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_row<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_row<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_adds_row_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_row_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_row_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_row_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_adds col tensor_Dense//
  if (test_send_adds_col<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_col<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_col<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_col<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_adds_col_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_col_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_col_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_col_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_adds row_line tensor_Dense//
  if (test_send_adds_row_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_row_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_row_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_row_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_adds_row_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_row_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_row_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_row_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  // vector_adds col_line tensor_Dense//
  if (test_send_adds_col_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_col_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_col_line<monolish::tensor::tensor_Dense<double>, double>(
          M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_col_line<monolish::tensor::tensor_Dense<float>, float>(
          M, N, L, 1.0e-3) == false) {
    return 1;
  }

  if (test_send_adds_col_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_adds_col_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }
  if (test_adds_col_line_view<double>(M, N, L, 1.0e-6) == false) {
    return 1;
  }
  if (test_adds_col_line_view<float>(M, N, L, 1.0e-3) == false) {
    return 1;
  }

  return 0;
}
