#pragma once
#include<vector>
#include"common/monolish_common.hpp"
#include<omp.h>
#include<stdio.h>

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{
	namespace blas{

////////////vector////////////////////

//////////////////////////////////////////////////////
//  axpy
//////////////////////////////////////////////////////
		/**
		 * @brief double precision axpy: y = ax + y
		 * @param[in] alpha double precision scalar value
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
		 */
  		void axpy(const double alpha, const vector<double> &x, vector<double> &y);

//////////////////////////////////////////////////////
//  axpyz
//////////////////////////////////////////////////////
		/**
		 * @brief double precision axpyz: z = ax + y
		 * @param[in] alpha double precision scalar value
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
		 * @param[in] z double precision monolish vector
		 */
  		void axpyz(const double alpha, const vector<double> &x, const vector<double> &y, vector<double> &z);

//////////////////////////////////////////////////////
//  xpay
//////////////////////////////////////////////////////
		/**
		 * @brief double precision xpay: y = x + ay
		 * @param[in] alpha double precision scalar value
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
		 * @param[in] z double precision monolish vector
		 */
  		void xpay(const double alpha, const vector<double> &x, vector<double> &y);

//////////////////////////////////////////////////////
//  scal
//////////////////////////////////////////////////////
		/**
		 * @brief double precision scal: x = alpha * x
		 * @param[in] alpha double precision scalar value
		 * @param[in] x double precision monolish vector
		 */
  		void scal(const double alpha, vector<double> &x);

//////////////////////////////////////////////////////
//  dot
//////////////////////////////////////////////////////
		/**
		 * @brief double precision inner product (dot)
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
 		 * @return The result of the inner product product of x and y
		 */
  		double dot(const vector<double> &x, const vector<double> &y);

		/**
		 * @brief float precision inner product (dot)
		 * @param[in] x float precision monolish vector
		 * @param[in] y float precision monolish vector
 		 * @return The result of the inner product product of x and y
		 */
  		float dot(const vector<float> &x,const vector<float> &y);

//////////////////////////////////////////////////////
//  nrm2
//////////////////////////////////////////////////////
		/**
		 * @brief double precision nrm2: ||x||_2
		 * @param[in] x double precision monolish vector
 		 * @return The result of the nrm2
		 */
  		double nrm2(const vector<double> &x);

//////////////////////////////////////////////////////
//  spmv (crs)
//////////////////////////////////////////////////////
		/**
		 * @brief double precision sparse matrix and vector multiplication in CRS: y = Ax
		 * @param[in] A double precision CRS matrix
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
		 */
		void spmv(matrix::CRS<double> &A, vector<double> &x, vector<double> &y);

	}
}
