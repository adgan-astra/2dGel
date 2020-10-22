CC=mpicc
OBJECTS=test2d.o mgsolver.o
CFLAGS=-Wall
LIBS= -lz 

  LIBS +=  /udrive/student/bnagda2015/silo-4.8/lib/libsilo.a -lm
  LIBS += -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
  LIBS += -lflapack -lfblas -lgfortran
  LIBS += -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib 

2dgel: test2d.o mgsolver.o
	$(CC) $(OBJECTS) -o $@  $(LIBS) -lm 

test2d.o: test2d.c
mgsolver.o: mgsolver.c gel.h

tagsfile:
	ctags -t *.h *.c
tarfile:
	tar -cvf gel.tar *.h *.c makefile 2dinput

clean:
	rm -f mgsolver.o test2d.o
	rm -f Makelog