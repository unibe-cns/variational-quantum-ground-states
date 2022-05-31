CXX=g++
CPPFLAGS=-O3 -std=c++11 -Wall -Wpedantic -Wextra
CPPFLAGS_DEBUG=-ftrapv
LDFLAGS=
LDLIBS=-lm -lyaml-cpp
INCLUDEPATH=-I/opt/homebrew/include
LIBPATH=-L/opt/homebrew/lib
#/opt/homebrew/lib

LATEXEXE=lualatex
MV=mv

TESTCXX=$(CXX) $(INCLUDEPATH) $(LIBPATH) $(LDFLAGS) $(CPPFLAGS)
OCXX=$(CXX) $(INCLUDEPATH) $(CPPFLAGS)

OBJS=neuralsampler network.o neuron.o

bin: bin/neuralsampler

all: bin test doc

test: tests/test_fixed_queue tests/test_neuron tests/test_config tests/test_network
	tests/test_fixed_queue
	tests/test_neuron
	tests/test_config
	tests/test_network

doc: doc/pdf/TSP.pdf

clean:
	$(RM) build/*
	$(RM) bin/*
	$(RM) test/*
	$(RM) src/*.gch
	$(RM) *.tmp

doc/pdf/TSP.pdf:
	$(LATEXEXE) -output-directory=tmp doc/tex/TSP/main.tex
	$(MV) tmp/main.pdf doc/pdf/TSP.pdf
	$(RM) tmp/main.aux tmp/main.log


build/network.o: src/network.cpp src/network.h
	$(OCXX) -c src/network.cpp -o build/network.o

build/configOutput.o: src/configOutput.cpp src/configOutput.h src/main.h
	$(OCXX) -c src/configOutput.cpp -o build/configOutput.o

build/config.o: src/config.cpp src/config.h src/configOutput.h    src/main.h
	$(OCXX) -c src/config.cpp -o build/config.o



bin/neuralsampler: src/main.cpp src/main.h src/myrandom.h build/config.o build/configOutput.o build/network.o
	$(CXX) $(INCLUDEPATH) $(LIBPATH) $(LDFLAGS) $(CPPFLAGS) build/config.o build/configOutput.o build/network.o src/main.cpp $(LDLIBS) -o bin/neuralsampler

