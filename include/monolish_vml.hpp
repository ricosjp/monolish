#pragma once
#include "common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
/**
 * @brief
 * Basic Linear Algebra Subprograms for Dense Matrix, Sparse Matrix, Vector and
 * Scalar
 */
namespace vml {

void add(const vector<double> &a, const vector<double> &b, vector<double> &y);
void sub(const vector<double> &a, const vector<double> &b, vector<double> &y);
void mul(const vector<double> &a, const vector<double> &b, vector<double> &y);
void div(const vector<double> &a, const vector<double> &b, vector<double> &y);

void add(const vector<double> &a, const double alpha, vector<double> &y);
void sub(const vector<double> &a, const double alpha, vector<double> &y);
void mul(const vector<double> &a, const double alpha, vector<double> &y);
void div(const vector<double> &a, const double alpha, vector<double> &y);


void add(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);
void sub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);
void mul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);
void div(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

void add(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);
void sub(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);
void mul(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);
void div(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);

void add(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);
void sub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);
void mul(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);
void div(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

void add(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);
void sub(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);
void mul(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);
void div(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

void add(const vector<float> &a, const vector<float> &b, vector<float> &y);
void sub(const vector<float> &a, const vector<float> &b, vector<float> &y);
void mul(const vector<float> &a, const vector<float> &b, vector<float> &y);
void div(const vector<float> &a, const vector<float> &b, vector<float> &y);

void add(const vector<float> &a, const float alpha, vector<float> &y);
void sub(const vector<float> &a, const float alpha, vector<float> &y);
void mul(const vector<float> &a, const float alpha, vector<float> &y);
void div(const vector<float> &a, const float alpha, vector<float> &y);

void add(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);
void sub(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);
void mul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);
void div(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);

void add(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);
void sub(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);
void mul(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);
void div(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);

void add(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);
void sub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);
void mul(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);
void div(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);

void add(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);
void sub(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);
void mul(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);
void div(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);
} // namespace blas
} // namespace monolish
