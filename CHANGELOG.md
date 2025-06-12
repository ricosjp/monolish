<!--
Please Keep this comment on the top of this file

How to write Changelog
-----------------------

https://keepachangelog.com/ja/1.0.0/ に基づいて記述していく

- Merge Request毎に記述を追加していく
- 何を変更したかを要約して書く。以下の分類を使う
  - Added      新機能について。
  - Changed    既存機能の変更について。
  - Deprecated 間もなく削除される機能について。
  - Removed    今回で削除された機能について。
  - Fixed      バグ修正について。
  - Security   脆弱性に関する場合。
- 日本語でも英語でも良い事にする

-->
Unreleased
-----------

0.17.2 - 2023/10/18
-----------

### Added
- Add new types: view_matrix / view_tensor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/504> <https://github.com/ricosjp/monolish/issues/731>
- Add tests for view_tensor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/505> <https://github.com/ricosjp/monolish/issues/735>


### Fixed
- Fix benchmark result token <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/502> <https://github.com/ricosjp/monolish/issues/733>
- Fix set_ptr bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/508> <https://github.com/ricosjp/monolish/issues/741>
- Fix linear solver bug for non-symmetric matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/510> <https://github.com/ricosjp/monolish/issues/750>

### Changed
- Update cuda version of allgebra <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/506> <https://github.com/ricosjp/monolish/issues/739>
- Change view1d constructor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/501> <https://github.com/ricosjp/monolish/issues/732>
- Change pyyaml version 5.4.1->5.3.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/507> <https://github.com/ricosjp/monolish/issues/740>
- Update node version 16->22 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/511> <https://github.com/ricosjp/monolish/issues/751>
- Update docker login variables <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/513> <https://github.com/ricosjp/monolish/issues/752>


0.17.1 - 2023/05/03
-----------

### Added
- Add matmul for C=aAB+bC <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/490> <https://github.com/ricosjp/monolish/issues/723>
- Add reshape for dense matrix  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/488> <https://github.com/ricosjp/monolish/issues/720>
- Add tensor (tensor_Dense & tensor_COO) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/492> <https://github.com/ricosjp/monolish/issues/724>
- Add adds_row, adds_col for dense <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/493> <https://github.com/ricosjp/monolish/issues/725>
- Add mattens functions as blas-like operation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/495> <https://github.com/ricosjp/monolish/issues/727>
- Add view1D of tensor_Dense <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/496> <https://github.com/ricosjp/monolish/issues/728>
- Add times/adds/axpy tests for view1D of matrix/tensor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/498> <https://github.com/ricosjp/monolish/issues/729>
- Add variadic templates for reshape tensor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/499> <https://github.com/ricosjp/monolish/issues/730>
- Add tensor_CRS <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/514> <https://github.com/ricosjp/monolish/issues/753>

### Fixed
- Fix convert (tensor-)coo to (tensor-)crs <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/516> <https://github.com/ricosjp/monolish/issues/755>

### Changed
- Start developing 0.17.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/487>
- std::vector -> std::shared_ptr for data array <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/491>
- Change operator[] as val.get()[] of value array <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/494> <https://github.com/ricosjp/monolish/issues/726>
- Change benchmark runner <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/489> <https://github.com/ricosjp/monolish/issues/722>

0.17.0 - 2022/11/15
-----------

### Fixed
- Add missings declaration in the include file of vml <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/485> <https://github.com/ricosjp/monolish/issues/718>

### Added

- Add max/min/alo for scalar-matrix/scalar-vector <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/482> <https://github.com/ricosjp/monolish/issues/713>
- Add exp for vector/Dense-matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/483> <https://github.com/ricosjp/monolish/issues/716>


### Changed
- Start developing 0.17.0 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/468>
- Use new token to push docker container into RICOS github container registry <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/484> <https://github.com/ricosjp/monolish/issues/714>
- Move include folder path  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/486> <https://github.com/ricosjp/monolish/issues/719>

0.16.3 - 2022/06/21
-----------

### Fixed
- fix diag_op bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/467> <https://github.com/ricosjp/monolish/issues/136>
- fix document typo <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/466> <https://github.com/ricosjp/monolish/issues/134>

### Changed
- Add compute_hash after transpose <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/465> <https://github.com/ricosjp/monolish/issues/135>
- Start developing 0.16.3 for fix bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/464>
- Start developing 0.17.0 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/463>

0.16.2 - 2022/06/15
-----------

### Added

- Add times for vector-matrix_line <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/472> <https://github.com/ricosjp/monolish/issues/124>
- Add times for vector-matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/471> <https://github.com/ricosjp/monolish/issues/124>
- Add times for scalar-matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/470> <https://github.com/ricosjp/monolish/issues/124>
- Add times for vector-vector <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/469> <https://github.com/ricosjp/monolish/issues/124>
- Add times for scalar-vector <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/469> <https://github.com/ricosjp/monolish/issues/124>
- Add CRS_diag_op <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/466> <https://github.com/ricosjp/monolish/issues/123>

### Fixed
- Add check line number in times_line <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/474> <https://github.com/ricosjp/monolish/issues/132>
- Support clang sm86 target <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/473> <https://github.com/ricosjp/monolish/issues/83>
- Fix some warnings <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/468> <https://github.com/ricosjp/monolish/issues/128>

### Changed

- Update clang14.0.4->clang14.0.5 (allgebra 22.06.0) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/475> <https://github.com/ricosjp/monolish/issues/131>
- Update clang13.0.1->clang14.0.4 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/467> <https://github.com/ricosjp/monolish/issues/125>
- Organize Dense_diag_op <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/464>
- Start developing 0.16.2 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/463>

0.16.1 - 2022/05/27
-----------

### Changed

- Update document (write developpers memo) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/462>
- Update poetry.lock (poetry update) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/461>
- Update allgebra 20.05.1 -> 20.05.4 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/461>
  - Ubuntu 20.04 -> 22.04
  - CUDA 11.6.0 -> 11.7.1
  - gcc 10 -> 11
  - cmake 3.21 -> 3.23
  - OpenMPI 4.1.0 -> 4.1.3
- Start developing 0.16.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/460>

0.16.0 - 2022/05/17
-----------

### Added

- Add Initialize constructors that create the same size matrix as input matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/458>
- Add internal::vbroadcast (y[i] = alpha) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/458>
- Add cmake option list in doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/457>
- Add specifing random seed in random vector/matrix generations <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/450>
- Add code generate scripts for transposed GEMM <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/448>
- Impl. transposed SpMV <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/447>
- Impl. asym_band_matrix() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/446>
- Impl. transposed gemv(matvec_N(Dense)) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/445> <https://github.com/ricosjp/monolish/issues/107>
- Impl. get_blas_trans_flag() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/444> <https://github.com/ricosjp/monolish/issues/107>
- Impl. matvec_N <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/442> <https://github.com/ricosjp/monolish/issues/107>
- Generate trans flag in matvec <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/442> <https://github.com/ricosjp/monolish/issues/107>
- Drop C trans (NNN->NN) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/441> <https://github.com/ricosjp/monolish/issues/107>
- Generate transpose matvec,matmul headers <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/439> <https://github.com/ricosjp/monolish/issues/107>

### Fixed

- Update and Fix document <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/449> <https://github.com/ricosjp/monolish/issues/116>
- Change parallel to sequential in random vector/matrix generations (for reproducibility) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/450>
- Change create_asym_band_matrix() in transposed GEMV <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/446>

### Changed

- Use cmake preset in Makefile <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/456> <https://github.com/ricosjp/monolish/issues/120>
- Update allgebra 21.02.0 -> 22.05.1 (integrate tools) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/455> <https://github.com/ricosjp/monolish/issues/119>
- Update allgebra 21.02.0 -> 22.05.0 (CUDA GPG key is updated) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/453> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/454> <https://github.com/ricosjp/monolish/issues/118>
- Change C++17 -> C++11 in headers, tests, and examples (user requiments changed to C++11) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/452>
- Specify random seed in test/ and examples/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/451>
- Skip zero elements in create_band_matrix() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/446>
- Change benchmark target matvec->matvec_N <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/445> <https://github.com/ricosjp/monolish/issues/107>
- Update allgebra 21.12.1 -> 22.02.0 (LLVM 13.0.0 -> 13.0.1) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/440> <https://github.com/ricosjp/monolish/issues/113>
- Start developing 0.16.0 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/438>

### Removed

- Drop transposed matrix-matrix operations (I will impl. 0.16.1) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/449>
- Drop SpMM/SpMV of CUDA 10.x support <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/443>

0.15.3 - 2022/01/13
-----------

### Added

- Refactoring documents <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/436> <https://github.com/ricosjp/monolish/issues/108>
- Add explanation of link_monolish_gpu.sh <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/434> <https://github.com/ricosjp/monolish/issues/102>
- Add get_final_resid() and get_final_iter() in equations and eigens <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/433> <https://github.com/ricosjp/monolish/issues/103>

### Fixed

- Fix LOBPCG warning <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/437> <https://github.com/ricosjp/monolish/issues/109>
- Fix logger gpu sync bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/433> <https://github.com/ricosjp/monolish/issues/106>
- Remove unnecessary function call <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/432> <https://github.com/ricosjp/monolish/issues/105>
- Fix logger bug in equations <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/432> <https://github.com/ricosjp/monolish/issues/104>

### Changed

- Start developing 0.15.3 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/431>

0.15.2 - 2021/12/26
-----------

### Fixed

- Fix monolish-container serious bug in master branch
- Fix serious monolish container bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/430>

### Changed

- Start developing 0.15.2 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/429>

0.15.1 - 2021/12/25
-----------

### Added

- Add explanation of IC and ILU in documents <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/425> <https://github.com/ricosjp/monolish/issues/94>
- Support sm_86 GPU <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/422>
- Add IC solver and preconditioner <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/416>
- Add ILU solver and preconditioner <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/411> <https://github.com/ricosjp/monolish/issues/42>
- Document for GitLab CI process <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/407>

### Fixed

- Fix warning of nodiscard in test/ and benchmark/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/428> <https://github.com/ricosjp/monolish/issues/96>
- Add GPU install guide <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/427>
- Fix Makefile bug by (<https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/423>) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/426>
- Add libomptarget.rtl.cuda.so in GPU deb package <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/420>
- Fix memory leak of IC/ILU precond <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/418>
- Fix calling create_precond() multiple times bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/417> <https://github.com/ricosjp/monolish/issues/93>
- Improve the perf of ILU solver and preconditioner <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/414> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/415>
- Improve perf of 2d5p Laplace matrix creation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/413>
- Fix 2d5p Laplace matrix creation unsorted bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/412>
- Remove `MONOLISH_DIR` variable in examples/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/408>

### Changed

- OpenMP runtimes are composed in libmonolish-common package <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/424>
- deb file split into shared library and others <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/423>
- Update LLVM13.0.0 in allgebra 21.12.0 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/421>
- Use allgebra 21.12.0 for monolish containers and CI <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/419>
- Rewrite quick start guide <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/406>
- Start developing 0.15.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/405>

0.15.0 - 2021/10/27
-----------

### Added

- Add CRS/Dense SOR precondition <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/380>
- Add CRS/Dense SOR solver <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/376>
- Create deb package by cpack <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/372>
- `MONOLISH_CCACHE_BUILD` flag to enable ccache <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/371>
- Add CMakePresets.json <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/368>
- Refactoring and add BiCGSTAB and Jacobi benchmarks in `benchmark/equation/` <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/365>
- Add `solver_name()` function <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/363>
- Collect iterative solver tests in `test/equation/iterative` <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/362>
- Add CRS.transpose() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/359> <https://github.com/ricosjp/monolish/issues/24>

### Fixed

- Rewrite int/size_t -> decltype() in src/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/398> <https://github.com/ricosjp/monolish/issues/87>
- Fix doxygen warnigs <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/394>
- Fix benchmark warnigs <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/393> <https://github.com/ricosjp/monolish/issues/80>
- Fix benchmark script bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/390>
- Fix memory leak of cusolver direct solver (getrf, sytrf) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/364>
- The Dense transpose() is not parallelized for non-square. <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/356>
- Fix bug, install gfortran in gitlab-CI <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/353>
- Fix typo in documents <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/351>

### Changed

- Installtion of monolish for GPU with apt does not support <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/403>
- Add nodiscard to solve() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/400>
- Version table in README <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/402>
- Add libomp and libomptarget in deb file <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/401>
- Change monolish container name oss_nvidia -> oss-nvidia, mkl_nvidia -> mkl-nvidia <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/398>
- Generate monolish container document by cmake <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/399>
- Use update-alternative to switch libmonolish_gpu.so <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/396>
- Split CMakeLists.txt into sub cmake script <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/395>
- Generate installation document by cmake <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/378>
- Move get_singularity and set_reorder in LU/QR/Cholesky to monolish::equation::solver <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/392> <https://github.com/ricosjp/monolish/issues/85>
- MONOLISH_PACKAGE_VERSION is added, SEND_ERROR if git command fails <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/391>
- Delete CUDA version in doxygen installation page <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/384>
- Update Doxyfile to 1.8.17, and disable warning for non-document contents <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/383>
- Support NVIDIA GPU CC=86 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/361> <https://github.com/ricosjp/monolish/issues/82>
- Use allgebra/poetry image for testing Python logger <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/360>
- Change transpose() function API <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/356> <https://github.com/ricosjp/monolish/issues/79>
  - `Dense Dense::transpose()` // return A^T -->  `void Dense::transpose()` // A=A^T
  - `void Dense::transpose(const Dense B)` // A=B^T, this function is not changed
  - `COO COO::transpose()` // return A^T --> `void COO::transpose()` // A=A^T
  - `void COO::transpose(COO B) const` // B=A^T -->  `void COO::transpose(const COO B)` // A=B^T
- OpenMP parallelized transpose() functions <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/356> <https://github.com/ricosjp/monolish/issues/79>
- Uses CUDA11 cuSPRASE SpMM <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/355> <https://github.com/ricosjp/monolish/issues/76>
- Remove nodiscard of COO.transpose() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/354> <https://github.com/ricosjp/monolish/issues/74>
- [support cuda11.x] cuda11.x cusparse generic API for CRS SpMV <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/352> <https://github.com/ricosjp/monolish/issues/65> <https://github.com/ricosjp/monolish/issues/75>
- [update allgebra 21.09.0] cuda 10.1 -> cuda11.4  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/352> <https://github.com/ricosjp/monolish/issues/65>
- [update allgebra 21.09.0] clang 12.0.0 -> clang12.0.1  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/352> <https://github.com/ricosjp/monolish/issues/65>

### Changed in CI

- Push to ghcr.io from GitLab CI <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/397>
- Generate all deb for specfic GPU arch <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/389>
- Generate deb for all GPU arch <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/386>
- Generate doxygen HTML and setup [view app] button for Merge Request <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/384>
- Fix AVX tag <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/385>
- Use MONOLISH_NVIDIA_GPU_ARCH while CI instead of detecting GPU <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/382>
- Fix CI setting of benchmark <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/381>
- Build CPU containers using cmake <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/379>
- Use cmake preset and deb package while CI <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/377>
- GitLab CI YAML is split into ci/*.yml <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/373>
- Revise GitHub Pages deployment <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/370>
- Revise Keep-Changelog CI job <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/369>
- monolish-docker stage when master --> when tags

0.14.2 - 2021/09/14
-----------

### Added

- add publications in doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/346>

### Added, Fixed, Changed (MPI trial implementation)

### Fixed

- fix OpenMP random bug in Dense and vector constructor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/350> <https://github.com/ricosjp/monolish/issues/70>
- fix typo random_stracture.cpp -> random_structure.cpp  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/349> <https://github.com/ricosjp/monolish/issues/71>
- fix deprecated MKL SpMV  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/344> <https://github.com/ricosjp/monolish/issues/65>
- delete unnecessary device free <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/344>
- delete gpu matrix transpose test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/344>

### Changed

- OpenMP parallelization for util::random_vector <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/350> <https://github.com/ricosjp/monolish/issues/70>
- update clang 11.0.0 --> 12.0.0 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/344> <https://github.com/ricosjp/monolish/issues/64>
- update allgebra 20.05.0--> 21.06.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/344> <https://github.com/ricosjp/monolish/issues/64>
- change std::copy --> internal::vcopy in transpose() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/344>

### Fixed

- Specify poetry version 1.1.7 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/345>
- stop MPI test (see <https://github.com/ricosjp/monolish/issues/67>) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/345>

0.14.1 - 2021/07/08
-----------

### Added

- add gitlab-CI test of examples/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/341>
- add only_cpu sample code <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/340>
- build monolish container latest in gitlab-CI <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/336>
- add CG, LU Benchmarks <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/333> <https://github.com/ricosjp/monolish/issues/63>
- add benchmark/ to monolish container <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/327> <https://github.com/ricosjp/monolish/issues/61>
- support NEC nlc lapack for SXAT <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/323>
- add matmul function for LinearOperator <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/322>
- add operator [] for matrix::Dense <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/320>
- add C++17 nodiscard attribute <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/318> <https://github.com/ricosjp/monolish/issues/58>
- add monolish::blas::sum test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/317>
- add jacobi preconditioner test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/311>
- add diag and Jacobi of LinearOperator <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/309>
- add Specifing GPU ID I/F <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/307>
- add testing and benchmarking document <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/305>
- add create laplacian 2D 5point <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/304>
- add cg/bicgstab test of equation in linearoerator <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/303>
- add monolish container <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/301>
- add build monolish_cpu on the local (and fix typo) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/300>
- add contribution approval flow <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/299>

### Added, Fixed, Changed (MPI trial implementation)

- fix test bug  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/342>
- change I/F of MPI non-blocking communication Isend/Irecv <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/332>
- delete gpu_sync flag <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/331>
- add MPI non-blocking communication Send/Recv <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/330>
- delete unnecessary val <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/329>
- change MPI::Comm --> MPI::comm <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/329>
- add MPI blocking communication Send/Recv <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/328>
- add MPI Bcast, Gather, Scatter <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/325>
- add MPI Barrier <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/318>
- add MPI gen scripts <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/318>
- add MPI sum(), asum(), nrm1(), nrm2() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/317> <https://github.com/ricosjp/monolish/issues/56>
- add MPI monolish::blas::dot() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/312>
- add get_rank(), get_size(), Allreduce() in monolish::mpi <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/312>
- add make install_mpi and make install_all <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/310>
- add monolish::mpi test in test/mpi <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/310>
- add monolish::mpi::Comm class <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/310>
- add MPI build test in gitlab-CI (beta) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/306>
- add build option `MONOLISH_USE_MPI` (beta) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/306>

### Fixed

- fix examples reference bug fix and clean makefile <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/339>
- change LU benchmark size <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/324>
- fix filename typo internal/*/interger.cpp --> integer.cpp <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/323>
- avoid SXAT c++17 std::random bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/323>
- test check_ans() bug fix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/316>
- test check_ans() bug fix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/316>
- LinearOperator diag() bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/313>
- generate documents for each tag (at github-actions, this work was done on master)
- Fix clang CPU build script bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/306>

### Changed

- change build flag MONOLISH_USE_GPU --> MONOLISH_USE_NVIDIA_GPU <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/342> <https://github.com/ricosjp/monolish/issues/62>
- add include sample code in doxygben <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/340>
- update monolish docker document <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/337>
- update monolish-log-viewer 0.1.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/335> <https://github.com/ricosjp/monolish/issues/59>
- support stop fcc trad mode, and support fcc clang mode <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/324>
- change C++17 simplified nested namespaces (namespace::namespace) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/321> <https://github.com/ricosjp/monolish/issues/60>
- change nrm2 --> dot() and sqrt for MPI <https://github.com/ricosjp/monolish/issues/57> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/316>
- change C++ version C++14 --> C++17 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/315>
- Use absolute tolerance when answer close to 0. <https://github.com/ricosjp/monolish/issues/53> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/314>
- update clang 11.0.0 --> 11.0.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/308>
- update allgebra 20.12.02-->21.05.0 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/308>
- organize doc/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/305>
- update solverlist for linearoerator <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/303>
- Fix update logger author <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/302>

### Removed

- del generate monolish container and doxygen at gitlab CI <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/301>

0.14.0 - 2021/04/05
-----------

### Added

- add build for users in doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/296>
- add solver table in doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/293>
- add gpu copy in doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/293>
- add solver.name() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/291>
- add samples in examples/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/290>
- add github pages for doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/288>
- add doxygen documents in english <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/286>
- add packaging setting in Makefile, CMakeLists.txt <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/285>

### Changed

- support benchmark for each archtecture <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/295>
- change "make in" of MONOLISH_INSTALL_DIR <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/292>

### Fixed

- fix clang warning, unnecessary variable <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/294>
- fix: bicgstab gpu bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/290>
- update readme <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/283>
- make clean to clean every generated files <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/287>
- Fix typo CMAKE_C_COMPILER -> CMAKE_CXX_COMPILER <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/289>
- check English in documentation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/298>
- Fix examples not to see MONOLISH_DIR <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/297>

0.13.0 - 2021/03/31
-----------

### Added

- add Dense diag operations for Dense <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/282>
- add row/col/diag functions for view1D <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/281>
- add LOBCPG for dense matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/278>
- add iterative solver for dense matrix <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/278>
- add create view1D from view1D <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/276>
- add vector utils in view1D <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/274>
- add get_residual_l2 in view1D <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/271>
- add VML in view1D<Dense> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/271>
- update readme <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/272>
- add output_mm in COO <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/265>
- support vector constructor from initializer_list <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/261>
- add set_ptr in all matrix dormat <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/262>
- add matrix constructor from monolish::vector  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/262>
- add matrix constructor from N-origin std::vector  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/262>

### Changed

- generate add2others in Dense diag operation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/282>
- vector utils to const <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/274>
- delete printf in transpose test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/268>
- move blas/matrix/getvec to util/ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/267>
- change char* to std::string in matrix I/O <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/265>
- change print_all() MM to COO (delete MM header) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/265>
- support input unsymmetric file MM format <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/265>

### Fixed

- fix sxat and a64fx makefile dirs <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/280>
- fix clang warnings <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/275>
- fix const non-const consistency <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/270>
- add checking gpu_status and size in getvec <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/267>
- do not test Frank matrix with LOBPCG since nonreliable <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/266>
- update operation list for doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/263>

### Removed

- delete return vector getvec function <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/267>
- delete row/col vector arithmetics in Dense <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/264>

0.12.0 - 2021/03/02
-----------

### Added

- add 1Dview fill <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/255>
- add build option information test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/254>
- add 1Dview VML <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/248>
- add benchmarking code in LOBPCG test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/250>
- add matvec of linearoperator with view1D <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/247>
- add view1D BLAS <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/245>
- add view1D axpy test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/245>

### Changed

- add scheme of initial vector handling in LOBPCG <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/257>
- add pragma once and shebang <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/251>
- generate matrix VML <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/248>
- support multiple eigenpairs in LOBPCG <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/249>
- Use view1D in LOBPCG <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/246>
- support view1D in is_same_size() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/245>

### Fixed

- fix view1D VML offset bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/260>
- fix to allow nvprof in docker <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/256>

### Removed

- delete std::vector row(), col(), diag() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/259>
- delete gpu cholesky <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/253>

0.11.1 - 2021/02/09
-----------

### Added

- add view1D hpp (cpp is not impl. only declareration) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/244>
- add Dense LU/Cholesky <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/240>
- add LOBPCG(Sparse) eigensolver for generalized eigenproblem <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/241>
- add DC(Dense) eigensolver for generalized eigenproblem <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/235>
- add test matrices for generalized eigenvalue problem from arxiv:2007.08130 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/235>
- define view1D class <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/233>
- add linearoperator util: is_XXX <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/227>
- add is_same_device_mem_stat <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/226>

### Changed

- change VML math functions template <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/243>
- change VML arithmetic functions template <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/239>
- change matrix BLAS template (without subvec_op and matmul) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/238>
- change vector BLAS template <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/237>
- change vecadd/vecsub template <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/236>
- change LOBPCG eigensolver using xpay, axpyz instead of scal, vecadd, and vecsub <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/230>
- change util error throw to assert using is_same_XX <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/231>
- change BLAS error throw to assert using is_same_XX <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/229>
- change linear operator VML error throw to assert using is_same_XX <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/228>
- organize linearoperator utils <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/227>
- change VML error throw to assert using is_same_XX <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/226>

### Fixed

- fix logger type inference <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/252>
- fix random CI failing of LOBPCG <https://gitlab.ritc.jp/ricos/monolish/-/jobs/97686>
- fix slow convergence bug in LOBPCG <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/230>
- fix Eigenvalue calculation routine of Frank, Tridiagonal Toeplitz <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/230>
- fix cublas handler leak in blas::scal <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/234>
- fix error check doxygen comment <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/232>
- fix LOBPCG fail with sygvd() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/225>

### Removed

- delete LinearOperator.copy <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/229>

0.11.0 - 2021/01/23
-----------

### Added

- LinearOperator <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/210>
- LOBPCG/DC works on GPU, too <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/220>
- add vector is_same_size, is_same_structure <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/224>
- add is_same_size <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/224>
- add variadic template is_same_structure <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/223>
- add is_same_structure <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/222>
- add create_hash and get_hash of matrix class <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/216>
- add internal::vhash <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/216>
- add Dense solver <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/214>
- add vector reciprocal test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/215>
- add VML max/min <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/215>

### Changed

- compute_hash in CRS convert and constructor <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/221>
- make internal syev()/sygv() interface to Fortran95-like <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/219>
- Summarize CI stage public function <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/218>
- comment out LOBPCG run_gpu <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/218>
- comment out vector print_all test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/218>
- Dense.set_nnz, set_row, set_col -> public function <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/217>
- move eigen -> standard_eigen namespace <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/213>
- move LAPACK raw functions to internal namespace <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/211>
- add lapack.h to repository and use LAPACK Fortran interface when using LAPACK internally. <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/212>

### Fixed

- fix get_hash and is_same_structure const <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/223>
- fix vhash return value bug <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/221>
- delete create_hash in COO and Dense <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/221>
- create_hash -> compute_hash <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/221>
- fix ans_check bug in test util <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/215>

0.10.0 - 2021/01/13
-----------

### Added

- add equal operation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/208>
- add CRS::convert(CRS) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/203>
- add Frank matrix, tridiagonal Toeplitz matrix, 1D Laplacian matrix as sample matrices <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/199>
- add blas::copy <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/201>
- add fill function <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/193>
- add util::build_with functions <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/192>
- add xpay test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/192>
- add nrm1 and get_residual_l2(Dense) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/191>
- add Frank matrix creation and eigenvalue calculation routine <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/189>
- add jacobi solver <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/188>
- add jacobi preconditioner <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/188>
- add vml::reciprocal <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/188>
- add LOBPCG eigensolver <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/88> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/194> <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/197> <https://gitlab.ritc.jp/ricos/monolish/-/issues/479>
- Support MKL SpMV and SpMM <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/185>
- install monolish_log_viewer on monolish container <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/183>
- add install-sxat install-a64fx target in makefile <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181>
- define four benchmarks {intel-MKL, intel-OSS, AMD-OSS, GPU-MKL} <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/179>
- add oss test <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178>
- add makefile target `make oss-cpu` `make oss-cpu` `make mkl-cpu` `make mkl-gpu` <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178>

### Changed

- operator=, != call equal operation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/208>
- organize src/util dir <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/198>
- change test-cpu -> test_cpu in makefile <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/198>
- monolish container created only web, tags, master <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/207>
- Drop FindMKL.cmake include guard <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/206>
- syev, sygv: automatically generate float from double using d2f.sh <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/204>
- change copy() to copy constructor in test dir <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/202>
- change operator= to blas::copy in equation <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/202>
- dont use direct reference solver class value in solve function <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/200>
- CRS.print_all() output matrixmarket format <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/196>
- support print_all() on GPU <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/196>
- move equation::solver and equation::precondition to solver::solver and solver::precondition <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/190>
- exclude src/internal doxygen <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/182>
- update allgebra 20.12.2 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/182>
- include algorithm in internal.hpp <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181>
- change name sx->sxat, fx->a64fx <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181>
- deploy benchmark only:master -> only:schedules(weekly) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/179>
- benchmark only:master -> only:schedules(weekly) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178>
- change CI job prefix name [ops]-[arch] -> [arch]-[ops] <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/178>

### Fixed

- fix specifications of copy, operator=, copy constructor, convert <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/203>
- fix LOBPCG iteration logic <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/195>
- fix sxat, a64fx makefile bugs <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/181>

### Removed

- delete vector.copy() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/202>
- delete Dense.copy() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/202>
- delete CRS.copy() <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/202>

0.9.1 - 2020/12/28
-----------

### Added

- add vecadd/vecsub doxygen comments <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/177>
- CG, BiCGSTABでBREAKDOWNしたりresidualがNaNになったときの判定を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/175>
- BiCGSTABの実装を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/174>

### Changed

- BiCGSTABを実装済としてDoxygenに反映, update doxygen project vesion to 0.9.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/177>
- CG,BiCGSTABでA,x,bのGPU Statusが一致していなければerrorになるようにした<https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/176>
- update allgebra 20.10.1 -> 20.12.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/173>
- CIのRunner指定をhostnameからGPUのsmタグに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/171>

### Fixed

- test/equationのminiterを0に設定 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/174>
- Doxygenのmarkdownのtableが崩れているのを修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/172>
- monolish_log_viewerの連続処理カウント処理を修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/169>

0.9.0 - 2020/12/21
-----------

### Added

- VMLのDoxygenコメントとmarkdownへの反映 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/166>
- monolish_log_viewer のライセンスを Apache-2.0 に設定 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/167>
- CRSに数学関数を追加 (min, max以外) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/164>
- Denseに数学関数を追加 (min, max以外) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/163>
- vectorに数学関数を追加 (min, max以外) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/162>
- internalに数学関数を追加 (min, max以外) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/160>
- Doxygenのfunction listにvmlに実装予定の数学関数の一覧を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/159>
- internalとvmlにvtanhのコードを実装 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/157>

### Changed

- internalのvdivをMKLに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/161>
- CIのRunner指定をホスト名でなくMacアドレスのタグ名に変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/158>
- test, benchmarkにvml::vtanhに変更．元の各クラスのメンバ関数としての実装は削除 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/157>
- testにscalar-matrixのVMLがなかったので追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/157>

### Fixed

- powerの乱数の範囲を1~2に変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/165>

0.8.0 - 2020/12/17
-----------

### Added

- VMLのDoxygenコメントを追加，Doxygenバージョンを0.8.0へ <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/156>
- matrixに一致判定関数を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/156>
- matrix四則演算関数を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/149>
- matadd/matsub関数を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/149>
- vecadd/vecsubを追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/148>
- vector四則演算関数を追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/147>
- matrix subを追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/146>

### Changed

- test/とbenchmarkをcommon, vml, blasの3つに分割して整理 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/151>
- 四則演算関数をmonolish::vml名前空間, src/vml/, include/monolish_vml.hppに移動 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/151>
- matadd/をmataddsub/に変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/149>
- test/benchmarkのvector_commonをoperatorでなく四則演算関数に変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/147>
- matrix add/subでdoubleからfloatを作るようにファイル構成を変更  <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/146>

### Removed

- すべてのクラスの四則演算のoperatorを削除, test,benchmarkも同様に削除 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/153>

0.7.1 - 2020/12/10
-----------

### Added

- matrix copyのテストを追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140>
- src/internalに配列に対する基本演算のコードを実装 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139>
- make testでmake test-cpuとmake-gpuを両方実行するようにした <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139>

### Changed

- vector,CRS,denseの現状をDoxygenに反映 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/145>
- cmake中の環境変数を MKL_ROOT から MKLROOT に変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/144>
- CRSのadd, copy, scalの裏側をinternalに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/142>
- denseのadd, copy, scalの裏側をinternalに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/141>
- CIのコンテナ名を変数にして上から再設定できるようにした(+gitlabから変数をRICOSのコンテナレジストリにした) <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/141>
- vectorのoperator==をCPU/GPU両方のデータの完全一致でtrueに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140>
- vector四則演算の裏側をinternalに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/140>
- ドキュメントの呼び出し関数と呼び出しライブラリ一覧を修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139>
- src/monolish_internal.hppをsrc/internal/monolish_internal.hppに移動 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/139>

0.7.0 - 2020/12/04
-----------

### Added

- cmakeでclang11+GPUに対応した <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/137>

### Changed

- CIのartifactの寿命を360分に延長 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138>
- benchmarkの高速化のために乱数値のベクトルでなく定数ベクトルに変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138>
- clang11gcc7コンテナに合わせてベンチマークサイズを変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138>
- monolishコンテナをclangでビルドするようにした <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/136>
- clangに合わせてtest/lang/fortranのオプションに-fPIEをつけるようにした <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/136>

### Removed

- cmake作成前に一時的に作成したMakefile.clang-gpuを削除 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/138>

0.6.2 - 2020/11/17
-----------

### Added

- benchmark結果のURLをDoxygenに記載 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/134>
- benchmarkで演算の種類(kind)を出力するようにした <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/129>

### Changed

- monolish_log_viewerにlintをかけた <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/135>
- benchmarkの出力ディレクトリ名をハッシュ名だけに戻した<https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/128>

### Fixed

- vectorのbenchmarkサイズと繰り返し回数をメモリエラーが起きない範囲に調整 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/131>
- vectorのbenchmarkがfailしてもCIでerror扱いにならないのを修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/131>

0.6.1 - 2020/11/15
-----------

### Changed

- benchmarkの出力ディレクトリ名変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/122>
- benchmarkの測定サイズ変更 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/121>
- benchmarkでサイズの繰り返しをヘッダでまとめで定義するようにした<https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/124>

### Added

- benchmarkでpipeline_idを出力するようにする <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/119>

### Fixed

- CG法のベクトルの更新がおかしいのを修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/127>
- ベンチマークの測定スクリプトのバグ修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/121>
- cusparseの関数が非同期で実行されているようなのでsyncを追加した <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/120>
- benchmarkをtagsとschedulesでは実行しないようにした <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/116>
- benchmarkの出力ファイルの末尾に半角スペースが入っているバグを修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/117>
- benchmarkの出力ファイルの末尾にtabが入っているバグを修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/115>
- matadd, mscalのbenchmarkの出力ファイルのバグを修正 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/114>

0.6.0 - 2020/11/04
-----------

### Added

- monolish_log_viewer にtest追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/118>
- ベンチマークのディレクトリを作成，masterでのみ実行するようにした <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/112>
- benchmark/にtsvを置くとCIの最後でベンチマーク結果としてアップロードされるようになった <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/111>
- version 0.0.4の `test/` を `benchmark/` として復活させた<https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/110>
- CHANGELOG.md (このファイル) の追加 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/106>
- GitLab CI で Merge Request 毎に origin/master から CHANGELOG.md に更新があるかチェックする <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/105>

### Changed

- `pyproject.toml` と `__init__.py` を作成して monolish_log_viewer を Python パッケージにする <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/107>
- タグ付けのときはkeep-changelogを走らないようにする<https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/113>
- cmakeでNVPTXのオプションに-misa=sm_35と-lmを付けた <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/109>
- test/logger/logging にある Python スクリプト群を Project TOP に移動させる <https://gitlab.ritc.jp/ricos/monolish/-/issues/325>
- loggerの3層以上のアルゴリズム改善 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/103>
- Base allgebra image switched from 20.10.0 to 20.10.1 <https://gitlab.ritc.jp/ricos/monolish/-/merge_requests/105>
