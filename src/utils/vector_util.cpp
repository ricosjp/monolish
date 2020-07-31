#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish{

	template<typename T>
		T util::get_residual_l2(matrix::CRS<T> &A, vector<T> &x, vector<T> &b){
			vector<T> tmp(x.size());
			tmp.send();

			blas::matvec(A,x,tmp); //tmp=Ax
			tmp = b - tmp;
			return blas::nrm2(tmp);
		}
	template double util::get_residual_l2(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
	template float util::get_residual_l2(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
}
