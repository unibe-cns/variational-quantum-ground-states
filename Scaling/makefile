all: neuralsampler/bin/neuralsampler neuralsampler/scripts/figure6.pdf

neuralsampler/bin/neuralsampler: neuralsampler/src/*
	cd neuralsampler && make

neuralsampler/scripts/figure6.pdf: neuralsampler/bin/neuralsampler neuralsampler/scripts/scaling.py neuralsampler/scripts/fig6.py neuralsampler/scripts/generate_sim.py
	cd neuralsampler/scripts && python scaling.py
	cd neuralsampler/scripts && python fig6.py
