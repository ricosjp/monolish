#include <iostream>
#include <typeinfo>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

#ifdef USE_GPU
#include "cuda_runtime.h"
#include "cusparse.h"
#endif

namespace monolish {

void blas::matvec(const matrix::CRS<double> &A, const vector<double> &x,
                  vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (A.get_row() != y.size() || A.get_col() != x.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  size_t m = A.get_row();
  size_t n = A.get_col();
  size_t nnz = A.get_nnz();
  const double *xd = x.data();
  double *yd = y.data();

  const double *vald = A.val.data();
  const int *rowd = A.row_ptr.data();
  const int *cold = A.col_ind.data();

#if USE_GPU // gpu

#if 0
#pragma acc data present(xd [0:n], yd [0:n], vald [0:nnz], rowd [0:n + 1],     \
                         cold [0:nnz])
#pragma acc parallel
		{
#pragma acc loop independent 
				for(int i = 0 ; i < n; i++){
					yd[i] = 0;
				}

#pragma acc loop independent
				for(int i = 0 ; i < n; i++){
					for(int j = rowd[i] ; j < rowd[i+1]; j++){
						yd[i] += vald[j] * xd[cold[j]];
					}
				}
		}
#else
  cusparseHandle_t sp_handle;
  cusparseCreate(&sp_handle);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

  const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

  const double alpha = 1.0;
  const double beta = 0.0;

#pragma acc data present(xd [0:n], yd [0:m], vald [0:nnz], rowd [0:m + 1],     \
                         cold [0:nnz])
#pragma acc host_data use_device(xd, yd, vald, rowd, cold)
  {
    check(cusparseDcsrmv(sp_handle, trans, m, n, nnz, &alpha, descr, vald, rowd,
                         cold, xd, &beta, yd));
  }

#endif

#else // cpu

		#pragma omp parallel for
		for(int i = 0 ; i < (int)A.get_row(); i++){
            double ytmp = 0.0;
			for(int j = A.row_ptr[i] ; j < A.row_ptr[i+1]; j++){
				ytmp += vald[j] * xd[A.col_ind[j]];
            }
            y[i] = ytmp;
        }

#endif

  logger.func_out();
}

void blas::matvec(const matrix::CRS<float> &A, const vector<float> &x,
                  vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (A.get_row() != y.size() || A.get_col() != x.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  size_t m = A.get_row();
  size_t n = A.get_col();
  size_t nnz = A.get_nnz();
  const float *xd = x.data();
  float *yd = y.data();

  const float *vald = A.val.data();
  const int *rowd = A.row_ptr.data();
  const int *cold = A.col_ind.data();

#if USE_GPU // gpu

#if 0
#pragma acc data present(xd [0:n], yd [0:n], vald [0:nnz], rowd [0:n + 1],     \
                         cold [0:nnz])
#pragma acc parallel
		{
#pragma acc loop independent 
				for(int i = 0 ; i < n; i++){
					yd[i] = 0;
				}

#pragma acc loop independent
				for(int i = 0 ; i < n; i++){
					for(int j = rowd[i] ; j < rowd[i+1]; j++){
						yd[i] += vald[j] * xd[cold[j]];
					}
				}
		}
#else
  cusparseHandle_t sp_handle;
  cusparseCreate(&sp_handle);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

  const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

  const float alpha = 1.0;
  const float beta = 0.0;

#pragma acc data present(xd [0:n], yd [0:m], vald [0:nnz], rowd [0:m + 1],     \
                         cold [0:nnz])
#pragma acc host_data use_device(xd, yd, vald, rowd, cold)
  {
    check(cusparseScsrmv(sp_handle, trans, m, n, nnz, &alpha, descr, vald, rowd,
                         cold, xd, &beta, yd));
  }

#endif

#else // cpu

		#pragma omp parallel for
		for(int i = 0 ; i < (int)A.get_row(); i++){
            double ytmp = 0.0;
			for(int j = A.row_ptr[i] ; j < A.row_ptr[i+1]; j++){
				ytmp += vald[j] * xd[A.col_ind[j]];
            }
            y[i] = ytmp;
        }

#endif

  logger.func_out();
}

template <typename T> vector<T> matrix::CRS<T>::operator*(vector<T> &vec) {
  vector<T> y(get_row());
  y.send();

  blas::matvec(*this, vec, y);

  return y;
}
template vector<double> matrix::CRS<double>::operator*(vector<double> &vec);
template vector<float> matrix::CRS<float>::operator*(vector<float> &vec);
} // namespace monolish
