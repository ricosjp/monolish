.PHONY: cpu gpu lib test 

all:cpu

cpu:
	make -j -f Makefile.cpu

gpu:
	make -j -f Makefile.gpu

lib:
	make -j -f Makefile.cpu libs

test:
	cd test; make -B

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 
