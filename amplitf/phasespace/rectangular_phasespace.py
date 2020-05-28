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

import math
import numpy as np
import tensorflow as tf
import amplitf.interface as atfi

class RectangularPhaseSpace:
    """
    Class for rectangular phase space in n dimensions
    """

    def __init__(self, ranges=((0., 1.))):
        """
        Constructor
        """
        self.ranges = ranges

    @atfi.function
    def inside(self, x):
        """
          Check if the point x is inside the phase space
        """
        inside = tf.constant([True], dtype=bool)
        for n, r in enumerate(self.ranges):
            var = self.coordinate(x, n)
            inside = tf.logical_and(inside, tf.logical_and(
                tf.greater(var, r[0]), tf.less(var, r[1])))
        return inside

    @atfi.function
    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    @atfi.function
    def unfiltered_sample(self, size, maximum = None):
        """
          Return TF graph for uniform sample of points within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [tf.random.uniform([size], r[0], r[1], dtype = atfi.fptype()) for r in self.ranges]
        if maximum is not None :
            v += [tf.random.uniform([size], 0., maximum, dtype = atfi.fptype())]
        return tf.stack(v, axis=1)

    @atfi.function
    def uniform_sample(self, size, maximum = None):
        """
          Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
          Note it does not actually generate the sample, but returns the data flow graph for generation,
          which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, maximum))

    @atfi.function
    def rectangular_grid_sample(self, sizes):
        """
          Create a data sample in the form of rectangular grid of points within the phase space.
          Useful for normalisation.
        """
        size = 1
        for i in sizes:
            size *= i
        v = []
        mg = np.mgrid[[slice(0, i) for i in sizes]]
        for i, (r, s) in enumerate(zip(self.ranges, sizes)):
            v1 = (mg[i]+0.5)*(r[1]-r[0])/float(s) + r[0]
            v += [v1.reshape(size).astype('d')]
        x = tf.stack(v, axis=1)
        return tf.boolean_mask(x, self.inside(x))

    @atfi.function
    def coordinate(self, sample, n):
        """
          Return coordinate number n from the input sample
        """
        return sample[..., n]

    def dimensionality(self):
        return len(self.ranges)

    def bounds(self):
        return list(self.ranges)
