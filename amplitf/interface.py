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

import tensorflow as tf
import numpy as np
import itertools
import sys

_fptype = tf.float64
_ctype = tf.complex128

function = tf.function(autograph=False, experimental_relax_shapes=True)
#function = tf.function(autograph=False)
#def function(f) : return f

def set_single_precision():
    """
      Set single floating point precision globally for AmpliTF package
    """
    global _fptype, _ctype
    _fptype = tf.float32
    _ctype = tf.complex64


def set_double_precision():
    """
      Set double floating point precision globally for AmpliTF package
    """
    global _fptype, _ctype
    _fptype = tf.float64
    _ctype = tf.complex128


def fptype():
    """
      Get global floating point type
    """
    global _fptype
    return _fptype


def ctype():
    """
      Get global complex type
    """
    global _ctype
    return _ctype

# _interface_dict maps AmpliTF functions to the 
# TF functions of the same syntax
_interface_dict = {
  "sum" : "tf.add_n",
  "abs" : "tf.abs",
  "max" : "tf.maximum",
  "min" : "tf.minimum",
  "conjugate" : "tf.conj",
  "real" : "tf.real",
  "imaginary" : "tf.imag",
  "sqrt": "tf.sqrt",
  "exp" : "tf.exp",
  "log" : "tf.math.log",
  "sin" : "tf.sin",
  "cos" : "tf.cos",
  "tan" : "tf.tan",
  "asin" : "tf.asin",
  "acos" : "tf.acos",
  "atan" : "tf.atan",
  "atan2" : "tf.atan2",
  "tanh" : "tf.tanh",
  "pow" : "tf.pow",
  "zeros" : "tf.zeros_like",
  "ones" : "tf.ones_like",
  "cross" : "tf.linalg.cross", 
  "reduce_max" : "tf.reduce_max", 
  "reduce_sum" : "tf.reduce_sum", 
  "reduce_mean" : "tf.reduce_mean", 
  "where" :  "tf.where", 
  "equal" :  "tf.equal", 
  "logical_and" : "tf.logical_and", 
  "logical_or" : "tf.logical_or", 
  "greater" : "tf.greater", 
  "less" :    "tf.less", 
}

# Load functions from _interface_dict into module locals()
m = sys.modules[__name__]
for k,v in _interface_dict.items() : 
  fun = exec(f"""
def {k}(*args) : 
  return {v}(*args)
  """)
  m.__dict__[k] = locals()[f"{k}"]

#@function
def complex(re, im) : 
    """
      Construct complex number from real and imaginary values
      
      :param re: Real value
      :param im: Imaginary value
      
      :returns: complex number
    """
    return tf.complex(re, im)

#@function
def cast_complex(re):
    """ 
      Cast a real value to complex type
      
      :param re: Real value
      :returns: Value of complex type with zero imaginary part
    """
    return tf.cast(re, dtype=ctype())

#@function
def cast_real(re):
    """ 
      Cast a number to real 

      :param re: Input value to be cast to real (e.g. integer)
      :returns: Real value
    """
    return tf.cast(re, dtype=fptype())

#@function
def const(c):
    """ 
      Declare constant of default floating poitn type
      
      :param c: Initial value
      :returns: Real constant value
    """
    return tf.constant(c, dtype=fptype())

#@function
def bool_const(c):
    """ 
      Declare constant of boolean type
    
      :param c: Initial value
      :returns: Boolean constant value
    """
    return tf.constant(c, dtype=bool)

#@function
def invariant(c):
    """ 
      Declare invariant. Invariant is the constant vector with the number of elements 
      corresponding to 1st index equal to one (e.g. array of shape (1, ...) ). 
      These represent the quantities which do not depend on "event", and can 
      be broadcast to event-dependent vectors. 
      
      :param c: Initial value
      :returns: Invariant constant value
      
    """
    return tf.constant([c], dtype=fptype())

#@function
def concat(x, axis = 0) : 
    """
      Concatenate a list of tensors along axis
      
      :param x: Input list of tensors
      :param axis: Concatenation axis
      :returns: Concatenated tensor
    """
    return tf.concat(x, axis = axis)

#@function
def stack(x, axis = 0) : 
    """
      Stack a list of tensors creating a new axis
      
      :param x: Input list of tensors
      :param axis: New axis index
      :returns: Stacked tensor
    """
    return tf.stack(x, axis = axis)

#@function
def pi():
    """
      Pi constant
      
      :returns: pi constant (3.14159265...)
    """
    return const(np.pi)

@function
def interpolate(t, c):
    """
      Multilinear interpolation on a rectangular grid of arbitrary number of dimensions
        :param t: TF tensor representing the grid (of rank N)
        :param c: Tensor of coordinates for which the interpolation is performed
        :returns: 1D tensor of interpolated values
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

def set_seed(seed):
    """
      Set random seed for AmpliTF package

      :seed: Initial random seed
    """
    tf.random.set_seed(seed)
