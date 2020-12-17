import amplitf.interface as atfi
import amplitf.dynamics as atfd

# Build cubic spline line shape with 5 knots
lineshape = atfd.build_spline_lineshape(3, [0., 1., 2., 3., 4.])

# Input data tensor
x = atfi.const([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.])

# Re and Im values of the amplitude in spline knots
ampl_re = [ 10., 20., 10., 30., 40.]
ampl_im = [ 1.,  2.,   6.,  6.,  6.]

# Print the values of the function
print( lineshape(x, ampl_re, ampl_im) )
