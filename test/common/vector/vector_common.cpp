#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

int main(int argc, char** argv){

	int size = atoi(argv[1]);

	std::random_device random;
  	monolish::vector<double> x(size, 123.0);

	// size()
	if(x.size() != size){ return 1; }

	//operator[] does not impl.
	if(x.data()[0] != 123.0){ return 1; }

	std::cout << "pass" << std::endl;

	return 0;
}
