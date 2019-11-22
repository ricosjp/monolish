.PHONY: cpu gpu lib test install

all:cpu

cpu:
	make -j -f Makefile.cpu

gpu:
	make -j -f Makefile.gpu

external:
	make -j -f Makefile.cpu libs

install:
	make -f Makefile.cpu install

test:
	cd test; make -B

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 
