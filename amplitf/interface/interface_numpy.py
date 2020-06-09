# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import itertools
import sys

_fptype = np.float64
_ctype = np.complex128

#function = tf.function(autograph=False, experimental_relax_shapes=True)
#function = tf.function(autograph=False)
def function(f) : return f
def generator_function(f) : return f

def set_single_precision():
    global _fptype, _ctype
    _fptype = np.float32
    _ctype = np.complex64


def set_double_precision():
    global _fptype, _ctype
    _fptype = np.float64
    _ctype = np.complex128


def fptype():
    global _fptype
    return _fptype


def ctype():
    global _ctype
    return _ctype


_interface_dict = {
  "sum" : "np.add",
  "abs" : "np.abs",
  "max" : "np.maximum",
  "min" : "np.minimum",
  #"complex" : "np.complex",
  "conjugate" : "np.conj",
  "real" : "np.real",
  "imaginary" : "np.imag",
  "sqrt": "np.sqrt",
  "exp" : "np.exp",
  "log" : "np.log",
  "sin" : "np.sin",
  "cos" : "np.cos",
  "tan" : "np.tan",
  "asin" : "np.arcsin",
  "acos" : "np.arccos",
  "atan" : "np.arctan",
  "atan2" : "np.arctan2",
  "tanh" : "np.tanh",
  "pow" : "np.pow",
  "zeros" : "np.zeros_like",
  "ones" : "np.ones_like",
  "cross" : "np.cross", 
  "reduce_max" : "np.amax", 
  "reduce_sum" : "np.sum", 
  "reduce_mean" : "np.mean", 
#  "stack" : "np.stack", 
  "where" : "np.where", 
  "equal" : "np.equal", 
  "logical_and" : "np.logical_and", 
  "greater" : "np.greater", 
  "less" :    "np.less", 
}

m = sys.modules[__name__]
for k,v in _interface_dict.items() : 
  fun = exec(f"""
def {k}(*args) : 
  return {v}(*args)
  """)
  m.__dict__[k] = locals()[f"{k}"]

def complex(re, im) : 
    return re + (0+1j)*im

def cast_complex(re):
    """ Cast a real number to complex """
    return re.astype(ctype())
    #return np.cast(re, dtype=ctype())

def cast_real(re):
    """ Cast a number to real """
    return re.astype(fptype())

def const(c):
    """ Declare constant """
    return np.array(c)

def bool_const(c):
    """ Declare constant """
    return np.array(c)

def invariant(c):
    """ Declare invariant """
    return np.array([c])

def concat(x, axis = 0) : 
    return np.concatenate(x, axis = axis)

def stack(x, axis = 0) : 
    return np.stack(x, axis = axis)

def pi():
    return const(np.pi)


'''
def interpolate(t, c):
    """
      Multilinear interpolation on a rectangular grid of arbitrary number of dimensions
        t : TF tensor representing the grid (of rank N)
        c : Tensor of coordinates for which the interpolation is performed
        return: 1D tensor of interpolated values
    """
    rank = len(t.get_shape())
    ind = tf.cast(tf.floor(c), tf.int32)
    t2 = tf.pad(t, rank*[[1, 1]], 'SYMMETRIC')
    wts = []
    for vertex in itertools.product([0, 1], repeat=rank):
        ind2 = ind + tf.constant(vertex, dtype=tf.int32)
        weight = tf.reduce_prod(
            1. - tf.abs(c - tf.cast(ind2, dtype=fptype())), 1)
        wt = tf.gather_nd(t2, ind2+1)
        wts += [weight*wt]
    interp = tf.reduce_sum(tf.stack(wts), 0)
    return interp
'''

def set_seed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)

def variable(x, shape, dtype) : 
    np.array(x)

def random_uniform(shape, minval, maxval) : 
    return np.random.uniform(minval, maxval, size = shape)

class FitParameter : 
    def __init__(self, name, init_value, lower_limit, upper_limit, step_size = 1e-6) :

        self.value = init_value
        self.init_value = init_value
        self.name = name
        self.step_size = step_size
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.prev_value = None
        self.fixed = False
        self.error = 0.
        self.positive_error = 0.
        self.negative_error = 0.
        self.fitted_value = init_value

    def assign(self, x) : 
        self.value = x

    def __call__(self) : 
        return self.value

    def update(self, value) : 
        if value != self.prev_value : 
            self.assign(value)
            self.prev_value = value

    def fix(self):
        self.fixed = True

    def float(self):
        self.fixed = False

    def setFixed(self, fixed):
        self.fixed = fixed

    def floating(self):
        """
          Return True if the parameter is floating and step size>0
        """
        return self.step_size > 0 and not self.fixed

    def numpy(self) : 
        return fptype()(self.value)

def create_gradient(nll, args, float_pars) :
    return False

def gradient(par) :
    return None

