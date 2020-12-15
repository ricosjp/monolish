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

  return 0;
}
