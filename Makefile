all: gendata get_arch_flags sort

CFLAGS=-O3 -std=c++11 -Xcompiler -fopenmp -Xcompiler -Wall
 
sort: sort.cu get_arch_flags
	nvcc $(CFLAGS) $(shell ./get_arch_flags) -o $@ $< -lgomp

gendata: gendata.cu get_arch_flags
	nvcc $(CFLAGS) $(shell ./get_arch_flags) -o $@ $< -lcurand

get_arch_flags: get_arch_flags.cu
	nvcc -O0 -o $@ $<

clean:
	rm -fv gendata get_arch_flags sort dump.bin log.txt
