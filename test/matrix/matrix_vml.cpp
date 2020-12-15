#include "vml/mm_add.hpp"
#include "vml/mm_sub.hpp"
#include "vml/mm_mul.hpp"
#include "vml/mm_div.hpp"

int main(int argc, char **argv) {

  if (argc != 4) {
    std::cout << "error!, $1:M, $2:N, $3:K" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t M = atoi(argv[1]);
  size_t N = atoi(argv[2]);
  size_t K = atoi(argv[3]);
  std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // mm_add Dense//
  if (test_send_mm_add<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_add<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_add<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_add<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_add CRS//
  if (test_send_mm_add<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_add<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_add<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_add<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_sub Dense//
  if (test_send_mm_sub<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_sub<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_sub<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_sub<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_sub CRS//
  if (test_send_mm_sub<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_sub<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_sub<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_sub<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_mul Dense//
  if (test_send_mm_mul<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_mul<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_mul<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_mul<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_mul CRS//
  if (test_send_mm_mul<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_mul<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_mul<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_mul<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_div Dense//
  if (test_send_mm_div<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_div<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_div<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_div<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_div CRS//
  if (test_send_mm_div<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_div<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_div<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_div<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
   return 1;
  }

  return 0;
}
