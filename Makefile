all:cpu

cpu:
	make -f Makefile.cpu

gpu:
	make -f Makefile.gpu

test:
	cd test; make; make run


clean:
	- rm lib/*
	- rm obj/*
