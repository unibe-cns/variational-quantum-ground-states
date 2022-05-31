
all: figure2.pdf figure3.pdf figure4.pdf figure4_cpu_comparison.pdf figure5.pdf figure6.pdf


figure6.pdf: timings.json fig6.py
	python fig6.py

figure5.pdf: fig5.py figure4.data data/run_nhidsweep_nhid20_reps10_noiseweight15_isyn500_lr1_W74F3_w.npy data/run_nhidsweep_nhid20_reps10_noiseweight15_isyn500_lr1_W74F3_dkl.npy figure5.data figure5_stability.data figure5_training.data figure5_discretization.data
	python fig5.py

figure4.pdf: fig4.py figure4.data rbm_fig4_200k.json rbm_fig4_200k_fid.json n_groundstate_energy.json
	python fig4.py

figure4_cpu_comparison.pdf: fig4_cpu_comparison.py figure4.data rbm_fig4_200k.json rbm_fig4_200k_fid.json n_groundstate_energy.json
	python fig4_cpu_comparison.py

figure3.pdf: fig3.py figure3.data figure3_run_emin_lowactivity_p.npy figure3_run_emin_highactivity_p.npy figure3_pmag_h0.1.data
	python fig3.py

figure2.pdf: fig2.py figure2_activation.data figure2_weights.npy
	python fig2.py