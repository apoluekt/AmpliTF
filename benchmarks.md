# Benchmarks

Based on running the tests/test_lb2dppi_fit.py script (100k data points, 400k normalisation points), 
in seconds per likelihood function call. These times do not include the time to compile the graphs. 
However, they represent the mixture of calls to only NLL evaluation and calls to NLL and its gradient. 

For conparison, the results obtained for the similar fit using TF1.3 and TensorFlowAnalysis are also given. However, the implementation is somewhat different (Minuit vs. IMinuit, different random generator for toy MC). 

RAM usage for this fit: around 8 Gb. VRAM on v100 was limited to 10 Gb. 

   * GPU server: Intel Xeon 2.1 GHz, 64 core + NVidia v100 GPU
      * CPU-only: 0.266 (autograph=False). ~43 threads in parallel on average
      * GPU: 0.060 (autograph=False)
      * GPU: 0.1172 (autograph=False, experimental_relax_shapes=True)
      * CPU-only, same fit with TF1.3 and TFA: 0.196
      * GPU, same fit with TF1.3 and TFA: 0.040
      
   * Laptop: Dell XPS, Intel Core i7-8550U 1.80 GHz, 8 threads. 
      * CPU-only: 0.857 (autograph=False)

   * Desktop: Intel Core i5-3570 3.40 GHz, 4 threads + NVidia GTX 750 Ti (2 Gb)
      * CPU-only: 0.736 (autograph=False)
      * GPU: 0.184 (autograph=False). W/o gradients (gradients do not fit into 2 Gb VRAM). 
