#include "vml/mm_add.hpp"
#include "vml/mm_div.hpp"
#include "vml/mm_mul.hpp"
#include "vml/mm_sub.hpp"
#include "vml/sm_add.hpp"
#include "vml/sm_div.hpp"
#include "vml/sm_mul.hpp"
#include "vml/sm_sub.hpp"

#include "vml/m_asin.hpp"
#include "vml/m_asinh.hpp"
#include "vml/m_atan.hpp"
#include "vml/m_atanh.hpp"
#include "vml/m_ceil.hpp"
#include "vml/m_floor.hpp"
#include "vml/m_max.hpp"
#include "vml/m_min.hpp"
#include "vml/m_reciprocal.hpp"
#include "vml/m_sign.hpp"
#include "vml/m_sin.hpp"
#include "vml/m_sinh.hpp"
#include "vml/m_sqrt.hpp"
#include "vml/m_tan.hpp"
#include "vml/m_tanh.hpp"
#include "vml/mm_max.hpp"
#include "vml/mm_min.hpp"
#include "vml/mm_pow.hpp"
#include "vml/sm_max.hpp"
#include "vml/sm_min.hpp"
#include "vml/sm_pow.hpp"

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

  // sm_add Dense//
  if (test_send_sm_add<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_add<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_add<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_add<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // sm_sub Dense//
  if (test_send_sm_sub<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_sub<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_sub<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_sub<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // sm_mul Dense//
  if (test_send_sm_mul<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_mul<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_mul<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_mul<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // sm_div Dense//
  if (test_send_sm_div<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_div<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_div<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_div<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // sm_pow Dense//
  if (test_send_sm_pow<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_pow<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_pow<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_pow<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_pow Dense//
  if (test_send_mm_pow<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_pow<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_pow<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_pow<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // msqrt Dense //
  if (test_send_msqrt<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msqrt<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msqrt<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msqrt<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // msin Dense //
  if (test_send_msin<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msin<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msin<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msin<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // msinh Dense //
  if (test_send_msinh<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msinh<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msinh<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msinh<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // masin Dense //
  if (test_send_masin<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_masin<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_masin<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_masin<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // masinh Dense //
  if (test_send_masinh<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_masinh<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_masinh<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_masinh<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mtan Dense //
  if (test_send_mtan<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mtan<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mtan<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mtan<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // matan Dense //
  if (test_send_matan<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_matan<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_matan<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_matan<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // matanh Dense //
  if (test_send_matanh<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_matanh<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_matanh<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_matanh<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mceil Dense //
  if (test_send_mceil<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mceil<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mceil<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mceil<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mfloor Dense //
  if (test_send_mfloor<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mfloor<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mfloor<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mfloor<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // msign Dense //
  if (test_send_msign<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msign<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msign<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msign<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mreciprocal Dense //
  if (test_send_mreciprocal<monolish::matrix::Dense<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_mreciprocal<monolish::matrix::Dense<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_mreciprocal<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mreciprocal<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mmax Dense //
  if (test_send_mmax<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mmax<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mmax<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mmax<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mmin Dense //
  if (test_send_mmin<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mmin<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mmin<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mmin<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_max Dense//
  if (test_send_mm_max<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_max<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_max<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_max<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_min Dense//
  if (test_send_mm_min<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_min<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_min<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_min<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // sm_max Dense//
  if (test_send_sm_max<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_max<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_max<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_max<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mm_min Dense//
  if (test_send_sm_min<monolish::matrix::Dense<double>,
                       monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_min<monolish::matrix::Dense<float>,
                       monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_min<monolish::matrix::Dense<double>,
                  monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_sm_min<monolish::matrix::Dense<float>,
                  monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  //////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////
  // CRS
  //////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////

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

  // sm_add CRS//
  if (test_send_sm_add<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_add<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_add<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_add<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_sub CRS//
  if (test_send_sm_sub<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_sub<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_sub<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_sub<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_mul CRS//
  if (test_send_sm_mul<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_mul<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_mul<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_mul<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_div CRS//
  if (test_send_sm_div<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_div<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_div<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_div<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_pow CRS//
  if (test_send_sm_pow<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_sm_pow<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_sm_pow<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_pow<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_pow CRS//
  if (test_send_mm_pow<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_pow<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_pow<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_pow<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // msqrt CRS //
  if (test_send_msqrt<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msqrt<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msqrt<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msqrt<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // msin CRS //
  if (test_send_msin<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msin<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msin<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_msin<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // msinh CRS //
  if (test_send_msinh<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msinh<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msinh<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msinh<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // masin CRS //
  if (test_send_masin<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_masin<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_masin<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_masin<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // masinh CRS //
  if (test_send_masinh<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_masinh<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_masinh<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_masinh<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mtan CRS //
  if (test_send_mtan<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mtan<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mtan<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_mtan<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // matan CRS //
  if (test_send_matan<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_matan<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_matan<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_matan<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // matanh CRS //
  if (test_send_matanh<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_matanh<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_matanh<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_matanh<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mceil CRS //
  if (test_send_mceil<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mceil<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mceil<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mceil<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mfloor CRS //
  if (test_send_mfloor<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mfloor<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mfloor<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mfloor<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // msign CRS //
  if (test_send_msign<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_msign<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_msign<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_msign<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mreciprocal CRS //
  if (test_send_mreciprocal<monolish::matrix::CRS<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_mreciprocal<monolish::matrix::CRS<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_mreciprocal<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mreciprocal<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }

  // mmax CRS //
  if (test_send_mmax<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mmax<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mmax<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_mmax<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mmin CRS //
  if (test_send_mmin<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mmin<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mmin<monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_mmin<monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_max CRS//
  if (test_send_mm_max<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_max<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_max<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_max<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_min CRS//
  if (test_send_mm_min<monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>,
                       monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_send_mm_min<monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>,
                       monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) ==
      false) {
    return 1;
  }
  if (test_mm_min<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
                  monolish::matrix::CRS<double>, double>(M, N, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_mm_min<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
                  monolish::matrix::CRS<float>, float>(M, N, 1.0e-4) == false) {
    return 1;
  }

  //////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////
  // LinearOperator
  //////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////

  // mm_add LinearOperator//
  if (test_send_mm_add<monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_mm_add<monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_mm_add<monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_mm_add<monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }

  // mm_sub LinearOperator//
  if (test_send_mm_sub<monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_mm_sub<monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_mm_sub<monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_mm_sub<monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_add LinearOperator//
  if (test_send_sm_add<monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_sm_add<monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_sm_add<monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_add<monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_sub LinearOperator//
  if (test_send_sm_sub<monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_sm_sub<monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_sm_sub<monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_sub<monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_mul LinearOperator//
  if (test_send_sm_mul<monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_sm_mul<monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_sm_mul<monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_mul<monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }

  // sm_div LinearOperator//
  if (test_send_sm_div<monolish::matrix::LinearOperator<double>,
                       monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_sm_div<monolish::matrix::LinearOperator<float>,
                       monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }
  if (test_sm_div<monolish::matrix::LinearOperator<double>,
                  monolish::matrix::LinearOperator<double>, double>(
          M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_sm_div<monolish::matrix::LinearOperator<float>,
                  monolish::matrix::LinearOperator<float>, float>(
          M, N, 1.0e-4) == false) {
    return 1;
  }

  return 0;
}
