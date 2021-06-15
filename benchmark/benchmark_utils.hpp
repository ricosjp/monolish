#include "monolish_blas.hpp"
#include "monolish_equation.hpp"
#include "monolish_vml.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>
#include <sstream>
#include <typeinfo>

// vector ///
#define VECTOR_BENCH_MIN 1000
#define VECTOR_BENCH_MAX 100000000
#define VECTOR_BENCH_INCL *= 10
#define VECTOR_BENCH_ITER 30

#define VECTOR_BLAS_BENCH_HEADER                                               \
  "func\tkind\tprec\tsize\ttime[sec]\tperf[GFLOPS]\tmem[GB/s]"

#define VECTOR_BLAS_OUTPUT_RESULT()                                            \
  std::cout << FUNC << "\t" << std::flush;                                     \
  std::cout << "vector"                                                        \
            << "\t" << std::flush;                                             \
  std::cout << get_type<T>() << "\t" << std::flush;                            \
  std::cout << size << "\t" << std::flush;                                     \
  std::cout << time << "\t" << std::flush;                                     \
  std::cout << PERF << "\t" << std::flush;                                     \
  std::cout << MEM << std::endl

// matrix ///
#define MATRIX_BENCH_ITER 30

// order N^2
#define DENSE_NN_BENCH_MIN 1000
#define DENSE_NN_BENCH_MAX 10000
#define DENSE_NN_BENCH_ITER += 1000

#define CRS_NN_BENCH_MIN 10000
#define CRS_NN_BENCH_MAX 100000
#define CRS_NN_BENCH_ITER += 10000

#define COO_NN_BENCH_MIN 1000
#define COO_NN_BENCH_MAX 10000
#define COO_NN_BENCH_ITER += 1000

// order N^3
#define DENSE_NNN_BENCH_MIN 1000
#define DENSE_NNN_BENCH_MAX 5000
#define DENSE_NNN_BENCH_ITER += 500

#define CRS_NNN_BENCH_MIN 1000
#define CRS_NNN_BENCH_MAX 3000
#define CRS_NNN_BENCH_ITER += 500

// LU, order 2/3*N^3
#define LU_BENCH_ITER 1
#define LU_NNN_BENCH_MIN 5000
#define LU_NNN_BENCH_MAX 30000
#define LU_NNN_BENCH_ITER += 5000

// CG
#define CG_BENCH_ITER 1
#define CG_NN_BENCH_MIN 500
#define CG_NN_BENCH_MAX 3000
#define CG_NN_BENCH_ITER += 500
#define CG_ITER 1000

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
