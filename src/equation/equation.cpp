#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish{


    template<typename T>
    T equation::solver<T>::get_residual(vector<T>& x){
        switch(resid_method){
            case 0:
                return blas::nrm2(x);
                break;
            default:
                throw std::runtime_error("error vector size is not same");
                break;
        }
    }

    template double equation::solver<double>::get_residual(vector<double>& x);
    template float equation::solver<float>::get_residual(vector<float>& x);
}
