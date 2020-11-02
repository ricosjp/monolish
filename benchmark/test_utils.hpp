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

template <typename T> bool ans_check(double result, double ans, double tol) {

  double err = std::abs(result - ans) / ans;

  if (err < tol) {
    return true;
  } else {
    std::cout << "Error!!" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << std::scientific << "result\t" << result << std::endl;
    std::cout << std::scientific << "ans\t" << ans << std::endl;
    std::cout << std::scientific << "Rerr\t" << err << std::endl;
    std::cout << "===============================" << std::endl;
    return false;
  }
}

template <typename T> bool ans_check(T *result, T *ans, int size, double tol) {

  std::vector<int> num;
  bool check = true;

  for (int i = 0; i < size; i++) {
    double err = std::abs(result[i] - ans[i]) / ans[i];
    if (err >= tol) {
      check = false;
      num.push_back(i);
    }
  }

  if (check) {
    return check;
  } else {
    std::cout << "Error!!" << std::endl;
    std::cout << "===============================" << std::endl;
    for (int i = 0; i < num.size(); i++) {
      std::cout << std::fixed << std::resetiosflags(std::ios_base::floatfield)
                << num[i] << "\tresult:" << std::flush;
      std::cout << std::fixed << std::setprecision(15) << result[num[i]]
                << "\tans:" << ans[num[i]] << std::endl;
    }
    std::cout << "===============================" << std::endl;
    return check;
  }
}

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
