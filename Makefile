CONTAINER  := registry.ritc.jp/ricos/allgebra:0.1.0
.PHONY: cpu gpu gpu-debug lib test install

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
	#docker run -it -u $$(id -u):$$(id -g) --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 
	#docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 
	#docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 
	docker pull $(CONTAINER) 
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 

in-cpu:
	#docker run -it -u $$(id -u):$$(id -g) --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 
	#docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 
	docker pull $(CONTAINER) 
	docker run -it --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 

format:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		registry.ritc.jp/ricos/allgebra/clang-format:0.2.0 /usr/bin/check-format.sh

document:
	docker run -it --rm  \
		-u `id -u`:`id -g` \
		-v $(PWD):$(PWD)   \
		-w $(PWD)          \
		registry.ritc.jp/ricos/allgebra/doxygen:0.2.0 doxygen Doxyfile
