all:cpu

cpu:
	make -f Makefile.cpu

gpu:
	make -f Makefile.gpu

tst:
	cd test; make

clean:
	- rm lib/*
	- rm obj/*
