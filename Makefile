ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra
ALLGEBRA_CUDA  := cuda11_7
ALLGEBRA_CC    := clang13
ALLGEBRA_TAG   := 22.05.4
LLVM_DIR := 13.0.1

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: cpu gpu gcc_cpu clang_cpu clang_gpu test_cpu test_gpu install install_cpu install_gpu in format document

MONOLISH_DIR ?= $(HOME)/lib/monolish

all: cpu gpu

##########################################

cpu: gcc_cpu
gpu: clang_gpu
install: install_cpu install_gpu

gcc_cpu:
	cmake $(MONOLISH_TOP)\
		--preset=cpu-avx-none \
		-DCMAKE_CXX_COMPILER=g++ \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

clang_cpu:
	cmake $(MONOLISH_TOP) \
		--preset=cpu-avx-none \
		-DCMAKE_CXX_COMPILER=clang++ \
		-Bbuild_cpu
	cmake --build build_cpu -j `nproc`

clang_gpu:
	cmake $(MONOLISH_TOP) \
		--preset=gpu-avx-none \
		-DCMAKE_CXX_COMPILER=clang++ \
		-Bbuild_gpu
	cmake --build build_gpu -j `nproc`

clang_gpu_all:
	cmake $(MONOLISH_TOP) \
		--preset=gpu-avx-none \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DMONOLISH_NVIDIA_GPU_ARCH_ALL=ON
		-Bbuild_gpu_all \
	cmake --build build_gpu_all -j `nproc`

a64fx:
	$(MAKE) -B -j4 -f Makefile.a64fx

sxat:
	$(MAKE) -B -j -f Makefile.sxat

install_cpu: cpu
	cmake --build build_cpu --target install
	cmake --preset=package-common -DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR)
	cmake --build build/package-common --target install

install_gpu: gpu
	cmake --build build_gpu --target install
	cmake --preset=package-common -DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR)
	cmake --build build/package-common --target install

install_gpu_all: clang_gpu_all
	cmake --build build_gpu_all --target install
	cmake --preset=package-common -DCMAKE_INSTALL_PREFIX=$(MONOLISH_DIR)
	cmake --build build/package-common --target install

install_sxat: 
	$(MAKE) -B -j -f Makefile.sxat install

install_a64fx: 
	$(MAKE) -B -j -f Makefile.a64fx install

########################################

cpu_mpi: clang_cpu_mpi
gpu_mpi: clang_gpu_mpi
install_mpi: install_cpu_mpi install_gpu_mpi

clang_cpu_mpi:
	cmake $(MONOLISH_TOP)\
		--preset=cpu-avx-mpi \
		-DCMAKE_C_COMPILER=mpicc \
		-DCMAKE_CXX_COMPILER=mpic++ \
		-Bbuild_cpu_mpi
	cmake --build build_cpu_mpi -j `nproc`

clang_gpu_mpi:
	cmake $(MONOLISH_TOP)\
		--preset=gpu-avx-mpi \
		-DCMAKE_C_COMPILER=mpicc \
		-DCMAKE_CXX_COMPILER=mpic++ \
		-Bbuild_gpu_mpi
	cmake --build build_gpu_mpi -j `nproc`

clang_gpu_mpi_all:
	cmake $(MONOLISH_TOP)\
		--preset=gpu-avx-mpi \
		-DCMAKE_C_COMPILER=mpicc \
		-DCMAKE_CXX_COMPILER=mpic++ \
		-DMONOLISH_NVIDIA_GPU_ARCH_ALL=ON \
		-Bbuild_gpu_mpi_all 
	cmake --build build_gpu_mpi_all -j `nproc`

install_cpu_mpi: cpu_mpi
	cmake --build build_cpu_mpi --target install

install_gpu_mpi: gpu_mpi
	cmake --build build_gpu_mpi --target install

install_gpu_mpi_all: clang_gpu_mpi_all
	cmake --build build_gpu_mpi_all --target install

install_all: install_cpu install_gpu install_cpu_mpi install_gpu_mpi

##########################################

test:
	test_cpu
	test_gpu

test_cpu: install_cpu
	$(MAKE) -C test cpu
	$(MAKE) -C test run_cpu

test_gpu: install_gpu
	$(MAKE) -C test gpu
	$(MAKE) -C test run_gpu

clean:
	- rm -rf build*/
	- $(MAKE) -f Makefile.a64fx clean
	- $(MAKE) -f Makefile.sxat clean
	- $(MAKE) -C test/ clean

##########################################

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
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG) /usr/bin/check_format.sh

document:
	rm -rf $(PWD)/build/document $(PWD)/html
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		$(ALLGEBRA_IMAGE)/$(ALLGEBRA_CUDA)/$(ALLGEBRA_CC)/oss:$(ALLGEBRA_TAG) bash -c "cmake -Bbuild/document .; cmake --build build/document --target document"
	mv $(PWD)/build/document/html $(PWD)

