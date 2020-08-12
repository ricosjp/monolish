#include"monolish_blas.hpp"
#include <ios>
#include<iostream>
#include<iomanip>
#include<cmath>
#include<random>

template <typename T>
bool ans_check(double result, double ans, double tol){

	double err = std::abs(result - ans) / ans;

	if(err < tol){
		return true;
	}
	else{
		std::cout << "Error!!" << std::endl;
		std::cout << "===============================" << std::endl;
		std::cout << std::scientific << "result\t" << result << std::endl; 
		std::cout << std::scientific << "ans\t" << ans << std::endl; 
		std::cout << std::scientific << "Rerr\t" << err << std::endl; 
		std::cout << "===============================" << std::endl;
		return false;
	}
}


template <typename T>
bool ans_check(
		T* result,
	   	T* ans,
	   	int size,
	   	double tol){


	std::vector<T> num;
	bool check = true;

	for(int i =0; i < size; i++)
	{
		double err = std::abs(result[i] - ans[i]) / ans[i];
		if(err >= tol){
			check = false;
			num.push_back(i);
		}
	}

	if(check){
		return check;
	}
	else{
		std::cout << "Error!!" << std::endl;
		std::cout << "===============================" << std::endl;
		for(int i=0; i < num.size(); i++){
			std::cout << std::fixed << std::resetiosflags(std::ios_base::floatfield) << num[i] <<"\tresult:" << std::flush;
            std::cout << std::fixed << std::setprecision(15) << result[i] << "\tans:" << ans[i] << std::flush; 
            std::cout << std::fixed << std::scientific << std::abs(result[i]-ans[i])/ans[i] << std::endl; 
		}
		std::cout << "===============================" << std::endl;
		return check;
	}
}
