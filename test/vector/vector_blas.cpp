#include "blas/asum.hpp"
#include "blas/axpy.hpp"
#include "blas/axpyz.hpp"
#include "blas/copy.hpp"
#include "blas/dot.hpp"
#include "blas/nrm1.hpp"
#include "blas/nrm2.hpp"
#include "blas/scal.hpp"
#include "blas/sum.hpp"
#include "blas/xpay.hpp"

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[1]);
  std::cout << "size: " << size << std::endl;

  // copy//
  if (test_copy<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_copy<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_copy<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_copy<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // sum//
  if (test_sum<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_sum<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_sum<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_sum<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // asum//
  if (test_asum<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_asum<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_asum<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_asum<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // axpy//
  if (test_axpy<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_axpy<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_axpy<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_axpy<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // xpay//
  if (test_xpay<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_xpay<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_xpay<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_xpay<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // axpyz//
  if (test_axpyz<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_axpyz<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_axpyz<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_axpyz<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // dot//
  if (test_dot<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_dot<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_dot<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_dot<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // nrm1//
  if (test_nrm1<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_nrm1<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_nrm1<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_nrm1<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // nrm2//
  if (test_nrm2<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_nrm2<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_nrm2<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_nrm2<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // scal//
  if (test_scal<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_scal<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_scal<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_scal<float>(size, 1.0e-4) == false) {
    return 1;
  }

  return 0;
}
