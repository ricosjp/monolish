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
  		void axpy(const float alpha, const vector<float> &x, vector<float> &y);

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
  		void axpyz(const float alpha, const vector<float> &x, const vector<float> &y, vector<float> &z);

//////////////////////////////////////////////////////
//  T dot 
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
//  void dot 
//////////////////////////////////////////////////////
		/**
		 * @brief double precision inner product (dot)
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
		 * @param[in] ans result value
		 */
  		void dot(const vector<double> &x, const vector<double> &y, double &ans);

		/**
		 * @brief float precision inner product (dot)
		 * @param[in] x float precision monolish vector
		 * @param[in] y float precision monolish vector
		 * @param[in] ans result value
		 */
  		void dot(const vector<float> &x,const vector<float> &y, float& ans);

//////////////////////////////////////////////////////
//  T nrm2
//////////////////////////////////////////////////////
		/**
		 * @brief double precision nrm2: ||x||_2
		 * @param[in] x double precision monolish vector
 		 * @return The result of the nrm2
		 */
  		double nrm2(const vector<double> &x);
  		float nrm2(const vector<float> &x);

//////////////////////////////////////////////////////
//  void nrm2
//////////////////////////////////////////////////////
		/**
		 * @brief double precision nrm2: ||x||_2
		 * @param[in] x double precision monolish vector
		 * @param[in] ans result value
		 */
  		void nrm2(const vector<double> &x, double& ans);
  		void nrm2(const vector<float> &x, float& ans);

//////////////////////////////////////////////////////
//  scal
//////////////////////////////////////////////////////
		/**
		 * @brief double precision scal: x = alpha * x
		 * @param[in] alpha double precision scalar value
		 * @param[in] x double precision monolish vector
		 */
  		void scal(const double alpha, vector<double> &x);
  		void scal(const float alpha, vector<float> &x);

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
  		void xpay(const float alpha, const vector<float> &x, vector<float> &y);

//////////////////////////////////////////////////////
//  spmv (crs)
//////////////////////////////////////////////////////
		/**
		 * @brief double precision sparse matrix and vector multiplication in CRS: y = Ax
		 * @param[in] A double precision CRS matrix
		 * @param[in] x double precision monolish vector
		 * @param[in] y double precision monolish vector
		 */
		void spmv(const matrix::CRS<double> &A, const vector<double> &x, vector<double> &y);
		void spmv(const matrix::CRS<float> &A, const vector<float> &x, vector<float> &y);

//////////////////////////////////////////////////////
//  matrix scale (crs)
//////////////////////////////////////////////////////
		/**
		 * @brief double precision scal: A = alpha * A
		 * @param[in] alpha double precision scalar value
		 * @param[in] A double precision CRS matrix
		 */
  		void mscal(const double alpha, matrix::CRS<double> &A);
  		void mscal(const float alpha, matrix::CRS<float> &A);
	}
}
