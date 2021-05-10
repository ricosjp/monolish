#include "../test_utils.hpp"
#include "monolish_blas.hpp"
#include <random>

template <typename T> bool test(const size_t size) {

  //(x) monolish::vector = std::vector  = 123, 123, ..., 123
  std::vector<T> std_vec_x(size, 123.0);
  monolish::vector<T> x(std_vec_x);

  //(y) monolish::vector = T* = 0,1,2, ..., N-1
  T *dp = (T *)malloc(sizeof(T) * size);
  for (size_t i = 0; i < size; i++) {
    dp[i] = i;
  }
  monolish::vector<T> y(dp, dp + size);

  //(z) monolish::vector.operator[] = 0,1,2, ..., N-1
  monolish::vector<T> z(size);
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = i;
  }

  monolish::vector<T> list = {1.0, 2.0, 3.0};

  // monolish::vector random(1.0~2.0) vector
  monolish::vector<T> randvec(size, 1.0, 2.0);

  // equal operator (z = rand(1~2)) on cpu
  z = randvec;

  // vec element add
  z[1] = z[1] + 111; // z[1] = rand(1~2) + 124 + 111 = 235+rand(1~2)

  // size check
  if (x.size() != size || y.size() != size || z.size() != size) {
    return false;
  }

  // gpu send and vec add
  monolish::vector<T> t(x.size());
  monolish::util::send(x, y, z, t);

  monolish::vml::add(x, y, t);
  monolish::vml::add(t, z, z); // rand(1~2) + 123+0, rand(1~2) + 123 + 111 ....
  if (monolish::util::build_with_gpu()) {
#if 0
    printf("gpu printf:\n");
    z.print_all(true); // printf test
#endif
  }
  if (monolish::util::build_with_gpu()) {
#if 0
    printf("cpu printf(before recv):\n");
    z.print_all(true); // printf test
#endif
  }

  // copy (cpu and gpu)
  monolish::vector<T> tmp = z;

  // recv vector z and tmp from gpu
  z.recv();
  tmp.recv();

  if (monolish::util::build_with_gpu()) {
#if 0
    printf("cpu printf(after recv):\n");
    z.print_all(); // printf test
#endif
  }

  // compare (on cpu)
  if (tmp != z) {
    std::cout << "error, copy fail " << std::endl;
    return false;
  }

  // range-based for statement (on cpu)
  for (auto &val : z) {
    val += 100; // z[1] = 336~337
  }

  if (!(336 < z[1] && z[1] < 337)) {
    std::cout << "error, z[1] = " << z[1] << std::endl;
    z.print_all();
    // z.print_all("./z.txt");
    return false;
  }

  std::cout << "Pass in " << __func__ << "(" << get_type<T>() << ")"
            << " precision" << std::endl;
  return true;
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  print_build_info();
  if (monolish::util::build_with_gpu) {
    std::cout << "# of GPU: " << monolish::util::get_num_devices() << std::endl;

    std::random_device rnd;
    int device = rnd() % monolish::util::get_num_devices();
    monolish::util::set_default_device(device);
    std::cout << "Setting default GPU number is " << device
              << " (random select)" << std::endl;
  }

  size_t size = atoi(argv[1]);

  if (size <= 1) {
    return 1;
  }

  // exec and error check
  if (test<double>(size) == false) {
    std::cout << "error in double" << std::endl;
    return 1;
  }

  if (test<float>(size) == false) {
    std::cout << "error in float" << std::endl;
    return 1;
  }
  return 0;
}
