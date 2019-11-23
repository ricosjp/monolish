CONTAINER  := registry.ritc.jp/ricos/allgebra
.PHONY: cpu gpu lib test install

all:cpu

cpu:
	make -B -j -f Makefile.cpu

gpu:
	make -B -j -f Makefile.gpu

external:
	make -j -f Makefile.cpu libs

install:
	make -f Makefile.cpu install

test:
	cd test; make -B

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 

in:
	docker run -it -u $$(id -u):$$(id -g) --gpus all --privileged --mount type=bind,src=$(PWD)/,dst=/monolish $(CONTAINER) 
