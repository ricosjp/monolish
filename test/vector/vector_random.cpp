#include "../test_utils.hpp"
#include "monolish_blas.hpp"
#include <random>
#include <algorithm>

template <typename T> bool test(size_t size) {

  //random
  monolish::vector<T> vec(size, 0, 100);

  std::vector<T> ans(size);
  for(size_t i=0;i<size;i++){
      ans[i] = vec[i];
  }

  std::sort(ans.begin(), ans.end());
  ans.erase(std::unique(ans.begin(), ans.end()), ans.end());

  if (ans.size() != size) {
    vec.print_all();
    std::cout << "error ans.size() = " << ans.size() << std::endl;
    return false;
  }

  std::cout << "random pass" << std::endl;

  //random+seed, seed=123
  monolish::vector<T> vec2(size, 0, 100, 123);

  std::vector<T> ans2(size);
  for(size_t i=0;i<size;i++){
      ans2[i] = vec2[i];
  }

  std::sort(ans2.begin(), ans2.end());
  ans.erase(std::unique(ans2.begin(), ans2.end()), ans2.end());

  if (ans2.size() != size) {
    vec2.print_all();
    std::cout << "error ans.size() = " << ans2.size() << std::endl;
    return false;
  }

  std::cout << "random+seed pass" << std::endl;

  //compare same seed
  monolish::vector<T> vec3(size, 0, 100, 123);

  if(vec2 != vec3){
    vec2.print_all();
    std::cout << "vec2!=vec3" << std::endl;
    return false;
  }
  std::cout << "compare random+seed pass" << std::endl;
  return true;

}

int main(int argc, char **argv) {

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = 10000;

  // exec and error check
  if (test<double>(size) == false) {
    std::cout << "error in double" << std::endl;
    return 1;
  }

  if (test<float>(size/100) == false) {
    std::cout << "error in float" << std::endl;
    return 1;
  }
  return 0;
}
