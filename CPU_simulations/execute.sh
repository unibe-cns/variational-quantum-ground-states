

for n in 3 4 5 6 7 8 9 10; do 
	python groundstate_learning.py --id review_N${n}_B1_Nh40_sampling_200k_lr001 --model TFIM --modelparams ${n} 1 1 --nhid 40 --lr 0.001 --epochs 10000  --optim ADAM --omitplots --record --sample_pvis --sample_grad --nsamples 20000 --nchains 10
done