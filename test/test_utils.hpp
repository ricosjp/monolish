#pragma once
#include "monolish_blas.hpp"
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

template <typename T> std::string get_type();
template <> std::string get_type<double>() { return "double"; }
template <> std::string get_type<float>() { return "float"; }

template <typename T>
bool ans_check(const std::string &func, double result, double ans, double tol) {

  double err = 1.0;
  if (std::abs(ans) <= tol) {
    err = std::abs(result - ans);
  } else {
    err = std::abs((result - ans) / ans);
  }

  if (err < tol) {
    std::cout << func << "(" << get_type<T>() << ")" << std::flush;
    std::cout << ": pass" << std::endl;
    return true;
  } else {
    std::cout << "Error!!" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << std::scientific << "result\t" << result << std::endl;
    std::cout << std::scientific << "ans\t" << ans << std::endl;
    std::cout << std::scientific << "Rerr\t" << err << std::endl;
    std::cout << "===============================" << std::endl;

    std::cout << func << "(" << get_type<T>() << ")" << std::flush;
    std::cout << ": fail" << std::endl;
    return false;
  }
}

template <typename T>
bool ans_check(const T *result, const T *ans, int size, double tol) {

  std::vector<int> num;
  bool check = true;

  for (int i = 0; i < size; i++) {
    double err = 1.0;
    if (ans[i] <= tol) {
      err = std::abs(result[i] - ans[i]);
    } else {
      err = std::abs((result[i] - ans[i]) / ans[i]);
    }
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

template <typename T>
bool ans_check(const std::string &func, const T *result, const T *ans, int size,
               double tol) {

  std::vector<int> num;
  bool check = true;

  for (int i = 0; i < size; i++) {
    double err = 1.0;
    if (std::abs(ans[i]) <= tol) {
      err = std::abs(result[i] - ans[i]);
    } else {
      err = std::abs((result[i] - ans[i]) / ans[i]);
    }
    if (err >= tol) {
      check = false;
      num.push_back(i);
    }
  }

  if (check) {
    std::cout << func << "(" << get_type<T>() << ")" << std::flush;
    std::cout << ": pass" << std::endl;
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
    std::cout << func << "(" << get_type<T>() << ")" << std::flush;
    std::cout << ": fail" << std::endl;
    return check;
  }
}

template <typename T>
bool ans_check(const std::string &func, const std::string &type,
               const T *result, const T *ans, int size, double tol) {

  std::vector<int> num;
  bool check = true;

  for (int i = 0; i < size; i++) {
    double err = 1.0;
    if (std::abs(ans[i]) <= tol) {
      err = std::abs(result[i] - ans[i]);
    } else {
      err = std::abs((result[i] - ans[i]) / ans[i]);
    }
    if (err >= tol) {
      check = false;
      num.push_back(i);
    }
  }

  if (check) {
    std::cout << func << "(" << get_type<T>() << "," << type << ")"
              << std::flush;
    std::cout << ": pass" << std::endl;
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
    std::cout << func << "(" << get_type<T>() << "," << type << ")"
              << std::flush;
    std::cout << ": fail" << std::endl;
    return check;
  }
}

void print_build_info() {

  if (monolish::util::build_with_avx()) {
    std::cout << "monolish: enable AVX" << std::endl;
  }

  if (monolish::util::build_with_avx2()) {
    std::cout << "monolish: enable AVX2" << std::endl;
  }

  if (monolish::util::build_with_avx512()) {
    std::cout << "monolish: enable AVX512" << std::endl;
  }

  if (monolish::util::build_with_gpu()) {
    std::cout << "monolish: enable GPU" << std::endl;
  }

  if (monolish::util::build_with_cblas()) {
    std::cout << "monolish: enable CBLAS" << std::endl;
  }

  if (monolish::util::build_with_mkl()) {
    std::cout << "monolish: enable MKL" << std::endl;
  }

  if (monolish::util::build_with_lapack()) {
    std::cout << "monolish: enable LAPACK" << std::endl;
  }
}
