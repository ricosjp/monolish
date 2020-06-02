#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish{

	 double equation::solver::get_residual(vector<double>& x){
		 switch(resid_method){
			 case 0:
				 return blas::nrm2(x);
				 break;
			 default:
				 throw std::runtime_error("error vector size is not same");
				 break;
		 }
	 }
	 float equation::solver::get_residual(vector<float>& x){
		 switch(resid_method){
			 case 0:
				 return blas::nrm2(x);
				 break;
			 default:
				 throw std::runtime_error("error vector size is not same");
				 break;
		 }
	 }
}
