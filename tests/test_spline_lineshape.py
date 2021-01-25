import amplitf.interface as atfi
import amplitf.dynamics as atfd

# Build cubic spline line shape with 5 knots
lineshape = atfd.build_spline_lineshape(3, [0.0, 1.0, 2.0, 3.0, 4.0])

# Input data tensor
x = atfi.const([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

# Re and Im values of the amplitude in spline knots
ampl_re = [10.0, 20.0, 10.0, 30.0, 40.0]
ampl_im = [1.0, 2.0, 6.0, 6.0, 6.0]

# Print the values of the function
print(lineshape(x, ampl_re, ampl_im))
