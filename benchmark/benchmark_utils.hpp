#include "monolish_blas.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>
#include <typeinfo>

#define VECTOR_BENCH_MIN 1000
#define VECTOR_BENCH_MAX 1000000

#define VECTOR_BLAS_BENCH_HEADER "func\tprec\tsize\ttime[sec]\tperf[GFLOPS]\tmem[GB/s]" 

#define VECTOR_BLAS_OUTPUT_RESULT() \
  std::cout << FUNC << "\t" << std::flush; \
  std::cout << get_type<T>() << "\t" << std::flush; \
  std::cout << size << "\t" << std::flush; \
  std::cout << time << "\t" << std::flush; \
  std::cout << PERF << "\t" << std::flush; \
  std::cout << MEM << "\t" << std::endl


template <typename T> std::string get_type() {
  std::string type;

  if (typeid(T) == typeid(double)) {
    type = "double";
  }
  if (typeid(T) == typeid(float)) {
    type = "float";
  }

  return type;
}

//
// template <typename Float_, typename Index_>
// static inline void
// make_3dSquare3PointsDirichlet_matrix(pzsparse::Matrix<Float_, Index_>& m,
// Index_ n)
// {
//     Index_ d = static_cast<Index_>(std::cbrt(static_cast<double>(n)));
//
//     m.reset_format(pzsparse::MatrixFormat_Uncompressed);
//     m.setzero();
//     m.resize(d * d * d);
//
//     for (Index_ i = 0; i < d; i++) {
//         for (Index_ j = 0; j < d; j++) {
//             for (Index_ k = 0; k < d; k++) {
//                 Index_ p = i * d * d + j * d + k;
//                 if (i == 0 || i == d - 1 || j == 0 || j == d - 1 | k == 0 ||
//                 k == d - 1) {
//                     m.insert(p, p, 1.0);
//                 } else {
//                     m.insert(p, p, -6.0);
//                     m.insert(p, i * d * d + j * d + (k + 1), 1.0);
//                     m.insert(p, i * d * d + j * d + (k - 1), 1.0);
//                     m.insert(p, i * d * d + (j + 1) * d + k, 1.0);
//                     m.insert(p, i * d * d + (j - 1) * d + k, 1.0);
//                     m.insert(p, (i + 1) * d * d + j * d + k, 1.0);
//                     m.insert(p, (i - 1) * d * d + j * d + k, 1.0);
//                 }
//             }
//         }
//     }
// }
