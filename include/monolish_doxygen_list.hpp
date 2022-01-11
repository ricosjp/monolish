/**
 * \defgroup Classes Basic Classes
 * @brief see @ref data_type page.
 * @{
 * \defgroup Vector_class monolsh::vector
 * @brief vector class.
 *
 * \defgroup View1D_class monolsh::view1D
 * @brief 1D view class.
 *
 * \defgroup Dense_class monolsh::matrix::Dense
 * @brief Dense format Matrix.
 *
 * \defgroup COO_class monolsh::matrix::COO
 * @brief Coodinate (COO) format Matrix.
 *
 * \defgroup CRS_class monolsh::matrix::CRS
 * @brief Compressed Row Storage (CRS) format Matrix.
 *
 * \defgroup LO_class monolsh::matrix::LinearOperator
 * @brief Linear Operator imitating Matrix.
 * @}
 */

/**
 * \defgroup Operations BLAS
 * @brief Basic Linear Algebra Subprograms for Dense Matrix, Sparse Matrix,
 * Vector and Scalar (see @ref oplist page).
 * @{
 * \defgroup BLASLV1 Vector Operations
 * @brief BLAS Lv1 vector operations.
 *
 * \defgroup BLASLV2 Matrix-Vector Operations
 * @brief BLAS Lv2 matrix-vector operations.
 *
 * \defgroup BLASLV3 Matrix-Matrix Operations
 * @brief BLAS Lv3 matrix-matrix operations.
 * @}
 */

/**
 * \defgroup VML VML
 * @brief Vector Math Library (VML) for Dense Matrix, Sparse Matrix, Vector and
 * Scalar (see @ref oplist page).
 * @{
 * \defgroup Vector_VML VML for Vector
 * @brief VML for vector.
 *
 * \defgroup Dense_VML VML for Dense
 * @brief VML for Dense matrix.
 *
 * \defgroup CRS_VML VML for CRS
 * @brief VML for CRS matrix.
 *
 * \defgroup LO_VML VML for LinearOperator
 * @brief VML for LinearOperator.
 * @}
 */

/**
 * \defgroup Solvers Solvers
 * @brief Linear equation solvers for Dense and sparse matrix (see @ref
 * solverlist page).
 * @{
 * \defgroup solver_base Base Class
 * @brief Solver base class.
 *
 * \defgroup equations Linear equations
 * @brief Linear equation solvers for Dense and sparse matrix.
 *
 * \defgroup sEigen Standard eigen
 * @brief Solve eigenvalues and eigenvectors problem.
 *
 * \defgroup gEigen Generalized eigen
 * @brief Solve generalized eigenvalues and eigenvectors problem.
 * @}
 */

/**
 * \defgroup utils Utilities
 * @brief Utilitie functions.
 * @{
 * \defgroup errcheck Check errors
 * @brief error check functions.
 *
 * \defgroup GPUutil Control GPU devices
 * @brief send, recv, and others..
 *
 * \defgroup logger Performance logger
 * @brief see @ref logger page.
 *
 * \defgroup build_options Get build options
 * @brief get status of build options (SIMD, BLAS, enable GPU device, etc.).
 *
 * \defgroup gendata Generate test data
 * @brief Generate test data.
 *
 * \defgroup compare Compare data
 * @brief Compare data.
 *
 * \defgroup Other Other
 * @brief Other utilitie functions.
 * @}
 */

/**
 * \defgroup MPI MPI class (beta)
 * @brief
 * C++ template MPI class, Functions of this class do nothing when MPI is
 * disabled.
 * Functions in this class are under development. Currently, Many BLAS functions
 * don't support MPI. Functions of this class does not support GPU. The user
 * needs to communicate with the GPU before and after the call to this function
 * if necessary.
 */
