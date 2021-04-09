# Compile and run simple program on CPU {#cpu_dev}

## Namespace and header files
The monolish namespace contains monolish::vector and monolish::view1D.
monolish has the following 7 namespaces in addition to the monolish namespace:
- monolish::matrix : Provides classes for Dense and Sparse matrices. This can be used by including other header files.
- monolish::blas : Provides a monolithic BLAS API that eliminates dependencies on data types, matrix format, and hardware-specific APIs. This is included in `monolish_blas.hpp`.
- monolish::vml : Provides the calculation of mathematical functions to each element of a vector. This is an open-source implementation of the [VML functions included in Intel MKL](https://software.intel.com/content/www/us/en/develop/articles/new-mkl-vml-api.html). This is included in `monolish_vml.hpp`.
- monolish::equation : Provides a solution for linear equations.	This is included in `monolish_equation.hpp`.
- monolish::standard_eigen : Provides solutions to standard eigenvalue problems. This is included in `monolish_eigen.hpp`.
- monolish::generalized_eigen : Provides a solution to the generalized eigenvalue problem. This is included in `monolish_generalized_eigen.hpp`.
- monolish::util : Provides utility functions. This can be used by including other header files.

This chapter describes a sample program using monolish that runs on the CPU.

Sample programs that run on the GPU can be found at `sample/`. They work on the CPU and GPU.
The programs running on the GPU are described in the next chapter.

## Compute innerproduct 
First, a simple inner product program is shown below:

\code{.cpp}
#include<iostream>
#include<monolish_blas.hpp>
int main(){

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");
  
  size_t N = 100;

  // x = {1,1,...,1}, length N
  monolish::vector<double> x(N, 1.0); 

  // Random vector length N with values in the range 1.0 to 2.0
  monolish::vector<double> y(N, 1.0, 2.0); 

  // compute innerproduct
  double ans = monolish::blas::dot(x, y); 

  std::cout << ans << std::endl;

  return 0;
}
\endcode

This program can be compiled by the following command.
```
g++ -O3 innerproduct.cpp -o innerproduct_cpu.out -lmonolish_cpu
```

The following command runs this.
``` 
./innerproduct_cpu.out
```

A description of this program is given below:
- A monolish::vector can be declared like a std::vector.
- As an extension, monolish::vector can also create random vectors.
- The inner product function is monolish::blas::dot(). It does not need type-dependent function names, for example `sdot` or `ddot`.
- For the BLAS library called inside monolish::blas::dot(), see [here](@ref oplist).
- At the end of the function, the memory of monolish::vector is automatically released by the destructor.

This program is executed in parallel. 
The number of threads is specified by the `OMP_NUM_THREADS` environment variable.

## Solve Ax=b
The following is a sample program that solves a linear equation; Ax=b using the conjugate gradient method with jacobi preconditioner.

\code{.cpp}
#include<iostream>
#include<monolish_equation.hpp>

int main(){

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  monolish::matrix::COO<double> A_COO("sample.mtx"); // Input from file

  // Edit the matrix as needed //
  // Execute A_COO.sort() after editing the matrix //

  monolish::matrix::CRS<double> A(A_COO); // Create CRS format and convert from COO format

  // length A.row()
  // Random vector length A.row() with values in the range 1.0 to 2.0
  monolish::vector<double> x(A.get_row(), 1.0, 2.0); 
  monolish::vector<double> b(A.get_row(), 1.0, 2.0); 

  // Create CG class
  monolish::equation::CG<monolish::matrix::CRS<double>, double> solver; 

  // Create jacobi preconditioner
  monolish::equation::Jacobi<monolish::matrix::CRS<double>, double> precond; 

  // Set preconditioner to CG solver
  solver.set_create_precond(precond); 
  solver.set_apply_precond(precond);

  // if you need residual history
  // solver.set_print_rhistory(true);
  // solver.set_rhistory_filename("./a.txt");

  // Set solver options
  solver.set_tol(1.0e-12);
  solver.set_maxiter(A.get_row()); 

  // Solve Ax=b by CG with jacobi
  monolish::util::solver_check(solver.solve(A, x, b)); 

  // Show answer
  x.print_all();
  
  return 0;
}
\endcode


### Matrix creation

monolish::matrix::COO has `editable` attribute, so users can edit any element using @ref monolish::matrix::COO.insert() "insert()" and monolish::matrix::COO.at() "at()" functions. COO can be created by giving array pointer or file name to the constructor, too.
This sample program creates a matrix from a file.
The input file format is [MatrixMarket format](https://math.nist.gov/MatrixMarket/formats.html).
MatrixMarket format is a common data format for sparse matrices. [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html#scipy.io.mmwrite) can also output matrices in MatrixMarket format.

monolish::matrix::CRS can easily be generated by taking COO as an argument to the constructor. 

### Solving Ax=b
monolish proposes a new way to implement linear solvers.

Applying preconditioning means combining multiple solvers. In other words, the solver and preconditioner in the Krylov subspace method are essentially the same thing.

The solver classes included in monolish::equation all have the following functions:
- @ref monolish::equation::CG.solve() "solve()" : Solve the simultaneous linear equations.
- @ref monolish::equation::CG.create_precond() "create_precond()" : Create the preconditioning matrix M.
- @ref monolish::equation::CG.apply_precond() "apply_precond()" : Apply preconditioner according to the created preprocessing matrix M.
- @ref monolish::equation::CG.set_create_precond() "set_create_precond()" : Register another class' create_precond to the class as a function to create the preprocessing matrix.
- @ref monolish::equation::CG.set_apply_precond() "set_apply_precond()" : Register "apply_precond" of another class to the class as a function to apply preprocessing.

The class that executes `solve()` execute the registered `create_precond()` and `apply_precond()`.
If no preconditioner is registered, it calls monolish::equation::none as "none preconditioner".

By being able to register the creation and application of preconditioners separately, users can use the preconditioner matrix in different ways.

In the current version, the create_precond() and apply_precond() functions of some classes do not work.
In the future, we will make these functions work in all classes.
This implementation would be very efficient for multi-grid methods.

See [here](@ref solverlist) for a list of solvers.

## Improve solver program
The program in the previous chapter has the matrix storage format, data type, solver name, and preconditioner name explicitly specified all over the program.

If users want to change them depending on the input matrix, this implementation requires a lot of program changes.

Since monolish is designed so that the matrix and solver classes all have the same interface, these changes can be eliminated by templating.

A templated program is shown below.

\code{.cpp}
#include<iostream>
#include"monolish_blas.hpp"
#include"monolish_equation.hpp"

// Template a matrix format, solver and preconditioer.
template<typename MATRIX, typename SOLVER, typename PRECOND, typename FLOAT>
void solve(){
  monolish::matrix::COO<FLOAT> A_COO("sample.mtx"); // Input from file

  // Edit the matrix as needed //
  // Execute A_COO.sort() after editing the matrix //
  
  MATRIX A(A_COO); // Create CRS format and convert from COO format

  // Length A.row()
  // Random vector length A.row() with values in the range 1.0 to 2.0
  monolish::vector<FLOAT> x(A.get_row(), 1.0, 2.0); 
  monolish::vector<FLOAT> b(A.get_row(), 1.0, 2.0); 

  // Create solver
  SOLVER solver; 

  // Create preconditioner
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  // Set solver options
  solver.set_tol(1.0e-12);
  solver.set_maxiter(A.get_row());

  // if you need residual history
  // solver.set_print_rhistory(true);
  // solver.set_rhistory_filename("./a.txt");

  // Solve
  monolish::util::solver_check(solver.solve(A, x, b)); 

  // output x to standard output
  x.print_all();
}

int main(){

  // output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  std::cout <<  "A is CRS, solver is CG, precondition is Jacobi, precision is double" << std::endl;
  solve<
    monolish::matrix::CRS<double>, 
    monolish::equation::CG<monolish::matrix::CRS<double>,double>,
    monolish::equation::Jacobi<monolish::matrix::CRS<double>,double>,
    double>();

  std::cout << "A is Dense, solver is BiCGSTAB, precondition is none, precision is float" << std::endl;
  solve<
    monolish::matrix::Dense<float>, 
    monolish::equation::BiCGSTAB<monolish::matrix::Dense<float>,float>,
    monolish::equation::none<monolish::matrix::Dense<float>,float>,
    float>();

  std::cout << "A is Dense, solver is LU, precondition is none, precision is double" << std::endl;
  solve<
    monolish::matrix::Dense<double>, 
    monolish::equation::LU<monolish::matrix::Dense<double>,double>,
    monolish::equation::none<monolish::matrix::Dense<double>,double>,
    double>();

  return 0;
}
\endcode

This program is templated so that the function that solves Ax=b is not oblivious to the matrix storage format or data type.
