#include"../test_utils.hpp"

template <typename T> bool test() {
  return true;
}

template <typename T> bool default_constructor_test() {

}

int main(int argc, char** argv){
  print_build_info();

  if(!test<double>()){
    return 1;
  }
  if(!test<float>()){
    return 1;
  }

  if(!default_constructor_test<double>()){
    return 2;
  }
  if(!default_constructor_test<float>()){
    return 2;
  }

  if(!fixed_size_test<double>()){
    return 3;
  }
  if(!fixed_size_test<float>()){
    return 3;
  }

  return 0;
}
