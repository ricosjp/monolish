#include "math/m_tanh.hpp"

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cout << "error!, $1:M, $2:N" << std::endl;
        return 1;
    }

    // monolish::util::set_log_level(3);
    // monolish::util::set_log_filename("./monolish_test_log.txt");

    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    std::cout <<  "M=" << M  << ", N=" << N << std::endl;

    // tanh Dense//
    if (test_send_tanh<
            monolish::matrix::Dense<double>, 
            double
            >(M, N, 1.0e-8) == false) {
        return 1;
    }
    if (test_send_tanh<
            monolish::matrix::Dense<float>, 
            float
            >(M, N, 1.0e-4) == false) {
        return 1;
    }
    if (test_tanh<
            monolish::matrix::Dense<double>, 
            double
            >(M, N, 1.0e-8) == false) {
        return 1;
    }
    if (test_tanh<
            monolish::matrix::Dense<float>, 
            float
            >(M, N, 1.0e-4) == false) {
        return 1;
    }

    // tanh CRS//
    if (test_send_tanh<
            monolish::matrix::CRS<double>, 
            double
            >(M, N, 1.0e-8) == false) {
        return 1;
    }
    if (test_send_tanh<
            monolish::matrix::CRS<float>, 
            float
            >(M, N, 1.0e-4) == false) {
        return 1;
    }
    if (test_tanh<
            monolish::matrix::CRS<double>, 
            double
            >(M, N, 1.0e-8) == false) {
        return 1;
    }
    if (test_tanh<
            monolish::matrix::CRS<float>, 
            float
            >(M, N, 1.0e-4) == false) {
        return 1;
    }

    return 0;
}
