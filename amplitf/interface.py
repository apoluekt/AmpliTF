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

#function = tf.function(autograph=False, experimental_relax_shapes=True)
function = tf.function(autograph=False)
#def function(f) : return f

def set_single_precision():
    global _fptype, _ctype
    _fptype = tf.float32
    _ctype = tf.complex64


def set_double_precision():
    global _fptype, _ctype
    _fptype = tf.float64
    _ctype = tf.complex128


def fptype():
    global _fptype
    return _fptype


def ctype():
    global _ctype
    return _ctype


_interface_dict = {
  "sum" : "tf.add_n",
  "abs" : "tf.abs",
  "max" : "tf.maximum",
  "min" : "tf.minimum",
  "complex" : "tf.complex",
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
}

m = sys.modules[__name__]
for k,v in _interface_dict.items() : 
  fun = exec(f"""
@function
def {k}(*args) : 
  return {v}(*args)
  """)
  m.__dict__[k] = locals()[f"{k}"]

@function
def density(ampl):
    """ density for a complex amplitude """
    return abs(ampl)**2

@function
def polar(a, ph):
    """ Create a complex number from a magnitude and a phase """
    return complex(a*cos(ph), a*sin(ph))

@function
def cast_complex(re):
    """ Cast a real number to complex """
    return tf.cast(re, dtype=ctype())

@function
def cast_real(re):
    """ Cast a number to real """
    return tf.cast(re, dtype=fptype())

@function
def const(c):
    """ Declare constant """
    return tf.constant(c, dtype=fptype())

@function
def invariant(c):
    """ Declare invariant """
    return tf.constant([c], dtype=fptype())

@function
def pi():
    return const(np.pi)

@function
def argument(c):
    """ Return argument (phase) of a complex number """
    return atan2(imag(c), real(c))

def clebsch(j1, m1, j2, m2, J, M):
    """
      Return clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
      (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
    """
    from sympy.physics.quantum.cg import CG
    from sympy import Rational
    return CG(Rational(j1, 2), Rational(m1, 2), Rational(j2, 2), Rational(m2, 2), Rational(J, 2), Rational(M, 2)).doit().evalf()

@function
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

def set_seed(seed):
    """
      Set random seed for numpy
    """
    tf.random.set_seed(seed)
