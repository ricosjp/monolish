#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

template <typename T>
T get_ans(monolish::vector<T> &mx, monolish::vector<T> &my){
	if(mx.size() != my.size()){
		std::runtime_error("x.size != y.size");
	}
	T ans = 0;
	T* x = mx.data();
	T* y = my.data();

	for(int i = 0; i < mx.size(); i++){
		ans += x[i] * y[i];
	}

	return ans;
}

template <typename T>
bool test(monolish::vector<T>& x, monolish::vector<T>& y, double tol, int iter, int check_ans){

	monolish::vector<T> tmp;
	tmp = y.copy();

	// check ans
	if(check_ans == 1){
		T result = monolish::blas::dot(x, y);
		T ans = get_ans(x, y);
		if(ans_check<T>(result, ans, tol) == false){
			return false;
		}
	}

	auto start = std::chrono::system_clock::now();

	for(int i = 0; i < iter; i++){
		std::cout << "iter:" << i << "\t" <<std::flush;
		T result = monolish::blas::dot(x, y);
	}

	auto end = std::chrono::system_clock::now();
	double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1.0e+9;

	std::cout << "total time: " << sec << std::endl;

	return true;
}

int main(int argc, char** argv){

	if(argc!=4){
		std::cout << "error $1:vector size, $2: iter, $3: error check (1/0)" << std::endl;
		return 1;
	}

	int size = atoi(argv[1]);
	int iter = atoi(argv[2]);
	int check_ans = atoi(argv[3]);

	std::random_device random;
  	monolish::vector<double> x(size, random());
  	monolish::vector<double> y(size, random());

	bool result;
	if( test<double>(x, y, 1.0e-8, iter, check_ans) == false){ return 1; }

	return 0;
}
