# Benchmarks

Based on running the tests/test_lb2dppi_fit.py script (100k data points, 400k normalisation points), 
in seconds per likelihood function call. These times do not include the time to compile the graphs. 
However, they represent the mixture of calls to only NLL evaluation and calls to NLL and its gradient. 

   * Intel Xeon 2.1 GHz, 64 core + NVidia v100 GPU
      * CPU-only: 0.333
      * GPU: 0.060 (autograph=False)
      * GPU: 0.1172 (autograph=False, experimental_relax_shapes=True)
 
