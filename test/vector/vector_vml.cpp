#include "vml/sv_add.hpp"
#include "vml/sv_div.hpp"
#include "vml/sv_mul.hpp"
#include "vml/sv_sub.hpp"
#include "vml/v_minus.hpp"
#include "vml/vecadd.hpp"
#include "vml/vecsub.hpp"
#include "vml/vv_add.hpp"
#include "vml/vv_div.hpp"
#include "vml/vv_mul.hpp"
#include "vml/vv_sub.hpp"

#include "vml/sv_pow.hpp"
#include "vml/v_asin.hpp"
#include "vml/v_asinh.hpp"
#include "vml/v_atan.hpp"
#include "vml/v_atanh.hpp"
#include "vml/v_ceil.hpp"
#include "vml/v_floor.hpp"
#include "vml/v_sign.hpp"
#include "vml/v_sin.hpp"
#include "vml/v_sinh.hpp"
#include "vml/v_sqrt.hpp"
#include "vml/v_tan.hpp"
#include "vml/v_tanh.hpp"
#include "vml/vv_pow.hpp"

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[1]);
  std::cout << "size: " << size << std::endl;

  // scalar-vetor-add//
  if (test_svadd<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_svadd<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_svadd<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_svadd<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // scalar-vetor-sub//
  if (test_svsub<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_svsub<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_svsub<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_svsub<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // scalar-vetor-mul//
  if (test_svmul<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_svmul<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_svmul<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_svmul<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // scalar-vetor-div//
  if (test_svdiv<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_svdiv<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_svdiv<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_svdiv<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vector-vetor-add//
  if (test_vvadd<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vvadd<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vvadd<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vvadd<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vector-vetor-sub//
  if (test_vvsub<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vvsub<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vvsub<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vvsub<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vector-vetor-mul//
  if (test_vvmul<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vvmul<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vvmul<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vvmul<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vector-vetor-div//
  if (test_vvdiv<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vvdiv<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vvdiv<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vvdiv<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vector-minus//
  if (test_minus<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_minus<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_minus<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_minus<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vecadd//
  if (test_vecadd<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vecadd<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vecadd<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vecadd<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vecsub//
  if (test_vecsub<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vecsub<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vecsub<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vecsub<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vvpow //
  if (test_vvpow<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vvpow<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vvpow<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vvpow<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // svpow //
  if (test_svpow<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_svpow<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_svpow<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_svpow<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vsqrt //
  if (test_vsqrt<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vsqrt<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vsqrt<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vsqrt<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vsin //
  if (test_vsin<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vsin<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vsin<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vsin<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vsinh //
  if (test_vsinh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vsinh<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vsinh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vsinh<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vasin //
  if (test_vasin<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vasin<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vasin<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vasin<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vasinh //
  if (test_vasinh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vasinh<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vasinh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vasinh<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vtan //
  if (test_vtan<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vtan<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vtan<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vtan<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vatan //
  if (test_vatan<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vatan<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vatan<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vatan<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vatanh //
  if (test_vatanh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vatanh<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vatanh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vatanh<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vceil //
  if (test_vceil<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vceil<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vceil<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vceil<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vfloor //
  if (test_vfloor<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vfloor<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vfloor<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vfloor<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // vsign //
  if (test_vsign<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vsign<float>(size, 1.0e-4) == false) {
    return 1;
  }
  if (test_send_vsign<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vsign<float>(size, 1.0e-4) == false) {
    return 1;
  }

  return 0;
}
