#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1:vector size" << std::endl;
		return 1;
	}

	int size = atoi(argv[1]);

	// monolish::vector = std::vector 
	std::vector<double> std_vec_x(size, 123.0);
  	monolish::vector<double> x(std_vec_x);

	// operator[]
	monolish::vector<double> y(size);
	for(size_t i=0; i<y.size(); i++){
		y[i] = i;
	}

	//constractor
  	monolish::vector<double> z(size, 0.0);


	// size()
	if(x.size() != size){ return 1; }

	//operator[] does not impl.
	if(x.data()[0] != 123.0){ return 1; }

	z = x + y;
	//std::cout << z.val[0] << std::endl;

	z.print_all();

	std::cout << z[0] << std::endl;
	z[0] = z[0] + 111;
	std::cout << z[0] << std::endl;

	std::cout << "pass" << std::endl;

	return 0;
}
	//std::random_device random;
