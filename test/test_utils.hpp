#include"monolish_blas.hpp"
#include<iostream>
#include<cmath>
#include<random>

template <typename T>
bool ans_check(double result, double ans, double tol){

	double err = std::abs(result - ans) / ans;

	if(err < tol){
		std::cout << "pass!!" << std::endl;
		return true;
	}
	else{
		std::cout << "Error!!" << std::endl;
		std::cout << "===============================" << std::endl;
		std::cout << "result\t" << result << std::endl; 
		std::cout << "ans\t" << ans << std::endl; 
		std::cout << "Rerr\t" << err << std::endl; 
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
		std::cout << "pass!!" << std::endl;
		return 0;
	}
	else{
		std::cout << "Error!!" << std::endl;
		std::cout << "===============================" << std::endl;
		for(int i=0; i < num.size(); i++){
			std::cout << num[i] <<"\tresult:" << result[i] << "\tans:" << ans[i] << std::endl; 
		}
		std::cout << "===============================" << std::endl;
		return 1;
	}
}

