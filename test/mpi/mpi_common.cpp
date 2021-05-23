#include "monolish_mpi.hpp"
int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }


  monolish::mpi::Comm &comm = monolish::mpi::Comm::get_instance();
  comm.Init(argc, argv);

  int rank = comm.get_rank();
  int size = comm.get_size();

  std::cout << "I am" << rank << "/" << size << std::endl;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[1]);
  std::cout << "size: " << size << std::endl;

  // nosend //
  if (test_dot<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_dot<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // send //
  if (test_dot<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_dot<float>(size, 1.0e-4) == false) {
    return 1;
  }

  comm.Finalize();

  return 0;
}
