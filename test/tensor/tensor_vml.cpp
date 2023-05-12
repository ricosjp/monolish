#include "vml/st_add.hpp"
#include "vml/st_div.hpp"
#include "vml/st_mul.hpp"
#include "vml/st_sub.hpp"
#include "vml/tt_add.hpp"
#include "vml/tt_div.hpp"
#include "vml/tt_mul.hpp"
#include "vml/tt_sub.hpp"

#include "vml/st_max.hpp"
#include "vml/st_min.hpp"
#include "vml/st_pow.hpp"
#include "vml/t_asin.hpp"
#include "vml/t_asinh.hpp"
#include "vml/t_atan.hpp"
#include "vml/t_atanh.hpp"
#include "vml/t_ceil.hpp"
#include "vml/t_exp.hpp"
#include "vml/t_floor.hpp"
#include "vml/t_max.hpp"
#include "vml/t_min.hpp"
#include "vml/t_reciprocal.hpp"
#include "vml/t_sign.hpp"
#include "vml/t_sin.hpp"
#include "vml/t_sinh.hpp"
#include "vml/t_sqrt.hpp"
#include "vml/t_tan.hpp"
#include "vml/t_tanh.hpp"
#include "vml/tt_max.hpp"
#include "vml/tt_min.hpp"
#include "vml/tt_pow.hpp"

#include "vml/t_alo.hpp"

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

  // tt_add tensor_Dense//
  if (test_send_tt_add<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_add<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_add<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_add<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_add_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_add_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_add_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_add_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tt_sub tensor_Dense//
  if (test_send_tt_sub<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_sub<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_sub<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_sub<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_sub_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_sub_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_sub_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_sub_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tt_mul tensor_Dense//
  if (test_send_tt_mul<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_mul<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_mul<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_mul<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_mul_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_mul_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_mul_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_mul_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tt_div tensor_Dense//
  if (test_send_tt_div<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_div<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_div<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_div<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_div_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_div_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_div_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_div_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_add tensor_Dense//
  if (test_send_st_add<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_add<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_add<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_add<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_add_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_add_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_add_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_add_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_sub tensor_Dense//
  if (test_send_st_sub<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_sub<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_sub<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_sub<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_sub_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_sub_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_sub_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_sub_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_mul tensor_Dense//
  if (test_send_st_mul<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_mul<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_mul<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_mul<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_mul_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_mul_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_mul_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_mul_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_div tensor_Dense//
  if (test_send_st_div<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_div<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_div<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_div<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_div_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_div_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_div_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_div_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_pow tensor_Dense//
  if (test_send_st_pow<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_pow<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_pow<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_pow<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_pow_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_pow_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_pow_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_pow_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tt_pow tensor_Dense//
  if (test_send_tt_pow<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_pow<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_pow<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_pow<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_pow_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_pow_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_pow_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_pow_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tsqrt tensor_Dense //
  if (test_send_tsqrt<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsqrt<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsqrt<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsqrt<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tsqrt_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsqrt_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsqrt_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsqrt_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tsin tensor_Dense //
  if (test_send_tsin<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsin<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsin<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsin<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tsin_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsin_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsin_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsin_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tsinh tensor_Dense //
  if (test_send_tsinh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsinh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsinh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsinh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tsinh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsinh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsinh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsinh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tasin tensor_Dense //
  if (test_send_tasin<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tasin<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tasin<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tasin<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tasin_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tasin_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tasin_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tasin_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tasinh tensor_Dense //
  if (test_send_tasinh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tasinh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tasinh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tasinh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tasinh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tasinh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tasinh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tasinh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // ttan tensor_Dense //
  if (test_send_ttan<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_ttan<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_ttan<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_ttan<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_ttan_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_ttan_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_ttan_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_ttan_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tatan tensor_Dense //
  if (test_send_tatan<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tatan<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tatan<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tatan<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tatan_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tatan_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tatan_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tatan_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // ttanh tensor_Dense //
  if (test_send_ttanh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_ttanh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_ttanh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_ttanh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_ttanh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_ttanh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_ttanh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_ttanh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tatanh tensor_Dense //
  if (test_send_tatanh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tatanh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tatanh<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tatanh<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tatanh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tatanh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tatanh_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tatanh_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tceil tensor_Dense //
  if (test_send_tceil<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tceil<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tceil<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tceil<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tceil_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tceil_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tceil_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tceil_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tfloor tensor_Dense //
  if (test_send_tfloor<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tfloor<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tfloor<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tfloor<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tfloor_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tfloor_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tfloor_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tfloor_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tsign tensor_Dense //
  if (test_send_tsign<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsign<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsign<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsign<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tsign_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tsign_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tsign_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tsign_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // treciprocal tensor_Dense //
  if (test_send_treciprocal<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_treciprocal<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_treciprocal<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_treciprocal<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_treciprocal_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_treciprocal_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_treciprocal_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_treciprocal_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // texp tensor_Dense //
  if (test_send_texp<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_texp<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_texp<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_texp<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_texp_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_texp_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_texp_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_texp_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tmax tensor_Dense //
  if (test_send_tmax<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tmax<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tmax<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tmax<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tmax_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tmax_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tmax_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tmax_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tmin tensor_Dense //
  if (test_send_tmin<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tmin<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tmin<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tmin<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tmin_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tmin_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tmin_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tmin_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tt_max tensor_Dense//
  if (test_send_tt_max<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_max<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_max<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_max<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_max_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_max_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_max_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_max_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // tt_min tensor_Dense//
  if (test_send_tt_min<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_min<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_min<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_min<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_tt_min_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_tt_min_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_tt_min_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_tt_min_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_max tensor_Dense//
  if (test_send_st_max<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_max<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_max<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_max<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_max_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_max_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_max_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_max_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // st_min tensor_Dense//
  if (test_send_st_min<monolish::tensor::tensor_Dense<double>,
                       monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_min<monolish::tensor::tensor_Dense<float>,
                       monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_min<monolish::tensor::tensor_Dense<double>,
                  monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_min<monolish::tensor::tensor_Dense<float>,
                  monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_st_min_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_st_min_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_st_min_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_st_min_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  // talo tensor_Dense //
  if (test_send_talo<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_talo<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_talo<monolish::tensor::tensor_Dense<double>, double>(
          M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_talo<monolish::tensor::tensor_Dense<float>, float>(
          M, N, K, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_talo_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_talo_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }
  if (test_talo_view<double>(M, N, K, 1.0e-8) == false) {
    return 1;
  }
  if (test_talo_view<float>(M, N, K, 1.0e-4) == false) {
    return 1;
  }

  return 0;
}
