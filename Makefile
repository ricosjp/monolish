ALLGEBRA_IMAGE := ghcr.io/ricosjp/allgebra/cuda10_2/mkl
ALLGEBRA_TAG   := 20.10.0

MONOLISH_TOP := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: cpu gpu gpu-debug lib test install in format

INSTALL_DIR=/usr/lib64/

all:cpu gpu

cpu:
	make -B -j -f Makefile.cpu

cpu-debug:
	make -B -j -f Makefile.cpu CXXFLAGS_EXTRA="-g3 -fvar-tracking-assignments"

gpu:
	make -B -j -f Makefile.gpu

gpu-debug:
	make -B -j -f Makefile.gpu CXXFLAGS_EXTRA="-g3 -fvar-tracking-assignments"

fx:
	make -B -j4 -f Makefile.fx

sx:
	make -B -j -f Makefile.sx

install:
	make -f Makefile.cpu install

test:
	cd test; make -B

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 
	- make -f Makefile.fx clean 
	- make -f Makefile.sx clean 
	- make -C test/ clean

zenbu:
	make clean
	make cpu
	make gpu
	make install

in:
	docker run -it --rm \
		--gpus all   \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE):$(ALLGEBRA_TAG)

in-cpu:
	docker run -it --rm \
		-e MONOLISH_DIR=/opt/monolish/0.1 \
		-e LD_LIBRARY_PATH=/opt/monolish/0.1/lib \
		-v $(MONOLISH_TOP):/monolish \
		-w /monolish \
		$(ALLGEBRA_IMAGE):$(ALLGEBRA_TAG)

format:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		ghcr.io/ricosjp/allgebra/clang-format:20.10.0 /usr/bin/check-format.sh

document:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		ghcr.io/ricosjp/allgebra/doxygen:20.10.0 doxygen Doxyfile
