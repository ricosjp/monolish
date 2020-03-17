#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include<iostream>

namespace monolish{
	 int equation::solver::precon_init(matrix::CRS<double>& A, vector<double>& p, vector<double>& q){

		switch(precon_num){
			case 0:
				break;
			case 1:
				equation::jacobi precon;
				precon.Pinit(A, p, q);
				break;
		}
		return 0;
	}
}
