#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include "../../../include/monolish_blas.hpp"

#define BENCHMARK
namespace monolish{

	template<>
	void vector<double>::add(const vector<double> &x, const vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");

		}
		
		for(size_t i = 0; i < size(); i++){
			val[i] = x.val[i] + y.val[i];
		}

	}

	template<>
	vector<double> vector<double>::operator+(vector<double>& vec){
		add(*this, vec);
		return *this;
	}

// 	vector operator+=(vector<Float>& vec);
// 	vector operator+(double value);

}
