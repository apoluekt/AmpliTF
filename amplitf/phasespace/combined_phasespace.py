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


class CombinedPhaseSpace:
    """
      Combination (direct product) of two phase spaces
    """

    def __init__(self, phsp1, phsp2):
        self.phsp1 = phsp1
        self.phsp2 = phsp2

    def dimensionality(self):
        return self.phsp1.dimensionality() + self.phsp2.dimensionality()

    @atfi.function
    def data1(self, x):
        return tf.slice(x, [0, 0], [-1, self.phsp1.dimensionality()])

    @atfi.function
    def data2(self, x):
        return tf.slice(x, [0, self.phsp1.dimensionality()], [-1, self.phsp2.dimensionality()])

    @atfi.function
    def inside(self, x):
        return tf.logical_and(self.phsp1.inside(self.data1(x)), self.phsp2.inside(self.data2(x)))

    @atfi.function
    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    @atfi.function
    def unfiltered_sample(self, size, maximum=None):
        """
          Return TF graph for uniform sample of point within phase space. 
            size     : number of _initial_ points to generate. Not all of them will fall into phase space, 
                       so the number of points in the output will be <size. 
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is 
                       uniform number from 0 to majorant. Useful for accept-reject toy MC. 
        """
        sample1 = self.phsp1.unfiltered_sample(size)
        sample2 = self.phsp2.unfiltered_sample(size, maximum)
        return tf.concat((sample1, sample2), axis=-1)

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

    def bounds(self):
        return self.phsp1.bounds() + self.phsp2.bounds()
