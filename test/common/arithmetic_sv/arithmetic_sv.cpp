#include<iostream>
#include<istream>
#include<chrono>
#include<cassert>
#include"../../test_utils.hpp"

template <typename T>
void get_ans(monolish::vector<T> &mx, T value, monolish::vector<T> &ans){
	if(mx.size() != ans.size()){
		std::runtime_error("x.size != y.size");
	}

	for(size_t i = 0; i < mx.size(); i++){
 		ans[i] += mx[i] + value;
  		ans[i] *= mx[i] * value;
  		ans[i] -= mx[i] - value;
 		ans[i] /= mx[i] / value;
	}
}

template <typename T>
bool test(monolish::vector<T>& x, T value, monolish::vector<T>& ans, double tol, const size_t check_ans){

	monolish::vector<T> ans_tmp;

	//copy
	ans_tmp = ans;
	if(ans_tmp[0] != 321.0 || ans_tmp.size() != ans.size()) {return false;}

	// check arithmetic
	if(check_ans == 1){
		ans += x + value;
		ans *= x * value;
		ans -= x - value;
		ans /= x / value;
 		get_ans(x, value, ans_tmp);
		ans.recv();
		if(ans_check<T>(ans.data(), ans_tmp.data(), x.size(), tol) == false){
 			return false;
 		}
	}

	return true;
}

int main(int argc, char** argv){

	if(argc!=3){
		std::cout << "error $1:vector size, $2: error check (1/0)" << std::endl;
		return 1;
	}
	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	size_t size = atoi(argv[1]);
	size_t check_ans = atoi(argv[2]);

	//create random vector x rand(0.1~1.0)
	double value = 123.0;
   	monolish::vector<double> x(size, 0.1, 1.0);
   	monolish::vector<double> ans(size, 321.0);
	x.send();
	ans.send();

 	// exec and error check
 	if( test<double>(x, value, ans, 1.0e-8, check_ans) == false){ return 1; }
	
	return 0;
}
