#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

template <typename T>
void get_ans(monolish::vector<T> &mx, monolish::vector<T> &my, monolish::vector<T> &mz, monolish::vector<T> &ans){
	if(mx.size() != my.size() || mx.size() != mz.size()){
		std::runtime_error("x.size != y.size");
	}

	for(size_t i = 0; i < mx.size(); i++){
 		ans[i] += mx[i] + my[i];
   		ans[i] *= mx[i] * my[i];
   		ans[i] -= mx[i] - my[i];
  		ans[i] /= mx[i] / my[i];
	}
}

template <typename T>
bool test(const size_t size, double tol, const size_t check_ans){


	//create random vector x rand(0.1~1.0)
   	monolish::vector<T> x(size, 0.1, 1.0);
   	monolish::vector<T> y(size, 0.1, 1.0);
   	monolish::vector<T> z(size, 0.1, 1.0);
   	monolish::vector<T> ans(size, 321.0);

	monolish::vector<T> ans_tmp;

	//copy
	ans_tmp = ans.copy();

	if(ans_tmp[0] != 321.0 || ans_tmp.size() != ans.size()) {return false;}

	// check arithmetic
	if(check_ans == 1){
 		get_ans(x, y, z, ans_tmp);

		monolish::util::send(x,y,z,ans);
		ans += x + y;
 		ans *= x * y;
 		ans -= x - y;
 		ans /= x / y;
		ans.recv();
		if(ans_check<T>(ans.data(), ans_tmp.data(), y.size(), tol) == false){
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

 	// exec and error check
 	if( test<double>(size, 1.0e-8, check_ans) == false){ return 1; }
 	// exec and error check
 	if( test<float>(size, 1.0e-5, check_ans) == false){ return 1; }

	return 0;
}
