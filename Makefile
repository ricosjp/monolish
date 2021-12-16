ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra
ALLGEBRA_CUDA  := cuda11_4
ALLGEBRA_CC    := clang12
ALLGEBRA_TAG   := 21.12.0

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: cpu gpu gcc_cpu clang_cpu clang_gpu test_cpu test_gpu install install_cpu install_gpu in format document

MONOLISH_DIR ?= $(HOME)/lib/monolish

all: cpu gpu

gcc_cpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/bin/gcc \
		-DCMAKE_CXX_COMPILER=/usr/bin/g++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

clang_cpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/local/llvm-12.0.1/bin/clang \
		-DCMAKE_CXX_COMPILER=/usr/local/llvm-12.0.1/bin/clang++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_cpu \
	cmake --build build_cpu -j `nproc`

clang_gpu:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/local/llvm-12.0.1/bin/clang \
		-DCMAKE_CXX_COMPILER=/usr/local/llvm-12.0.1/bin/clang++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu \
		-DMONOLISH_USE_NVIDIA_GPU=ON
	cmake --build build_gpu -j `nproc`

clang_gpu_all:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=/usr/local/llvm-12.0.1/bin/clang \
		-DCMAKE_CXX_COMPILER=/usr/local/llvm-12.0.1/bin/clang++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu_all \
		-DMONOLISH_USE_NVIDIA_GPU=ON \
		-DMONOLISH_NVIDIA_GPU_ARCH_ALL=ON
	cmake --build build_gpu_all -j `nproc`

cpu: gcc_cpu
gpu: clang_gpu

a64fx:
	$(MAKE) -B -j4 -f Makefile.a64fx

sxat:
	$(MAKE) -B -j -f Makefile.sxat

install: install_cpu install_gpu

install_cpu: cpu
	cmake --build build_cpu --target install

install_gpu: gpu
	cmake --build build_gpu --target install

install_gpu_all: clang_gpu_all
	cmake --build build_gpu_all --target install

install_sxat: 
	$(MAKE) -B -j -f Makefile.sxat install

install_a64fx: 
	$(MAKE) -B -j -f Makefile.a64fx install

########################################

clang_cpu_mpi:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=mpicc \
		-DCMAKE_CXX_COMPILER=mpic++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_cpu_mpi \
		-DMONOLISH_USE_NVIDIA_GPU=OFF \
		-DMONOLISH_USE_MPI=ON
	cmake --build build_cpu_mpi -j `nproc`

clang_gpu_mpi:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=mpicc \
		-DCMAKE_CXX_COMPILER=mpic++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu_mpi \
		-DMONOLISH_USE_NVIDIA_GPU=ON \
		-DMONOLISH_USE_MPI=ON
	cmake --build build_gpu_mpi -j `nproc`

clang_gpu_mpi_all:
	cmake $(MONOLISH_TOP) \
		-DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR) \
		-DCMAKE_C_COMPILER=mpicc \
		-DCMAKE_CXX_COMPILER=mpic++ \
		-DCMAKE_VERBOSE_MAKEFILE=1 \
		-Bbuild_gpu_mpi_all \
		-DMONOLISH_USE_NVIDIA_GPU=ON \
		-DMONOLISH_NVIDIA_GPU_ARCH_ALL=ON \
		-DMONOLISH_USE_MPI=ON
	cmake --build build_gpu_mpi_all -j `nproc`

cpu_mpi: clang_cpu_mpi
gpu_mpi: clang_gpu_mpi

install_mpi: install_cpu_mpi install_gpu_mpi

install_cpu_mpi: cpu_mpi
	cmake --build build_cpu_mpi --target install

install_gpu_mpi: gpu_mpi
	cmake --build build_gpu_mpi --target install

install_gpu_mpi_all: clang_gpu_mpi_all
	cmake --build build_gpu_mpi_all --target install

install_all: install_cpu install_gpu install_cpu_mpi install_gpu_mpi

##########################################

test_cpu: install_cpu
	$(MAKE) -C test cpu
	$(MAKE) -C test run_cpu

test_gpu: install_gpu
	$(MAKE) -C test gpu
	$(MAKE) -C test run_gpu

test:
	test_cpu
	test_gpu

clean:
	- rm -rf build*/
	- $(MAKE) -f Makefile.a64fx clean
	- $(MAKE) -f Makefile.sxat clean
	- $(MAKE) -C test/ clean

in_mkl_gpu:
	docker run -it --rm \
		--gpus all   \
		--cap-add SYS_ADMIN \
		-e MONOLISH_DIR=/opt/monolish/ \
		-e LD_LIBRARY_PATH=/opt/monolish/lib:/usr/local/lib/ \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/mkl:$(ALLGEBRA_TAG)

in_mkl_cpu:
	docker run -it --rm \
		-e MONOLISH_DIR=/opt/monolish/ \
		-e LD_LIBRARY_PATH=/opt/monolish/lib:/usr/local/lib/ \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/mkl:$(ALLGEBRA_TAG)

in_oss_gpu:
	docker run -it --rm \
		--gpus all   \
		--cap-add SYS_ADMIN \
		-e MONOLISH_DIR=/opt/monolish/ \
		-e LD_LIBRARY_PATH=/opt/monolish/lib:/usr/local/lib/ \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG)

in_oss_cpu:
	docker run -it --rm \
		-e MONOLISH_DIR=/opt/monolish/ \
		-e LD_LIBRARY_PATH=/opt/monolish/lib:/usr/local/lib/ \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG)

in_cpu: in_mkl_cpu
in_gpu: in_mkl_gpu
in: in_gpu

format:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		$(ALLGEBRA_IMAGE)/clang-format:$(ALLGEBRA_TAG) /usr/bin/check_format.sh

document:
	rm -rf $(PWD)/build/document $(PWD)/html
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG) bash -c "cmake -Bbuild/document .; cmake --build build/document --target document"
	mv $(PWD)/build/document/html $(PWD)

