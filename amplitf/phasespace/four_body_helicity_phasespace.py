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
import amplitf.kinematics as atfk

class FourBodyHelicityPhaseSpace:
    """
    Class for 4-body decay phase space D->(A1 A2)(B1 B2) expressed as:
      ma   : invariant mass of the A1 A2 combination
      mb   : invariant mass of the B1 B2 combination
      hela : cosine of the helicity angle of A1
      helb : cosine of the helicity angle of B1
      phi  : angle between the A1 A2 and B1 B2 planes in D rest frame
    """
    def __init__(self, ma1, ma2, mb1, mb2, md, ma1a2range=None, mb1b2range=None, costharange=None, costhbrange=None, 
                 mab1range = None, 
                 mab2range = None):
        """
          Constructor
        """
        self.ma1 = ma1
        self.ma2 = ma2
        self.mb1 = mb1
        self.mb2 = mb2
        self.md = md

        self.ma1a2min = self.ma1 + self.ma2
        self.ma1a2max = self.md - self.mb1 - self.mb2
        self.mb1b2min = self.mb1 + self.mb2
        self.mb1b2max = self.md - self.ma1 - self.ma2
        self.costhamin = -1.
        self.costhamax =  1.
        self.costhbmin = -1.
        self.costhbmax =  1.

        self.mab1range = mab1range
        self.mab2range = mab2range

        if ma1a2range:
            if ma1a2range[1] < self.ma1a2max:
                self.ma1a2max = ma1a2range[1]
            if ma1a2range[0] > self.ma1a2min:
                self.ma1a2min = ma1a2range[0]
        if mb1b2range:
            if mb1b2range[1] < self.mb1b2max:
                self.mb1b2max = mb1b2range[1]
            if mb1b2range[0] > self.mb1b2min:
                self.mb1b2min = mb1b2range[0]
        if costharange : 
            self.costhamin = costharange[0]
            self.costhamax = costharange[1]
        if costhbrange : 
            self.costhbmin = costhbrange[0]
            self.costhbmax = costhbrange[1]

    @atfi.function
    def simple_inside(self, x):
        """
          Check if the point x is inside the phase space
        """
        ma1a2 = self.m_a1a2(x)
        mb1b2 = self.m_b1b2(x)
        ctha = self.cos_helicity_a(x)
        cthb = self.cos_helicity_b(x)
        phi = self.phi(x)

        inside = tf.logical_and(tf.logical_and(tf.greater(ctha, self.costhamin), tf.less(ctha, self.costhamax)),
                                tf.logical_and(tf.greater(cthb, self.costhbmin), tf.less(cthb, self.costhbmax)))
        inside = tf.logical_and(inside,
                                tf.logical_and(tf.greater(
                                    phi, -math.pi), tf.less(phi, math.pi))
                                )

        mb1b2max = atfi.min(atfi.cast_real(self.mb1b2max), atfi.cast_real(self.md) - ma1a2)

        inside = tf.logical_and(inside, tf.logical_and(tf.greater(
            ma1a2, self.ma1a2min), tf.less(ma1a2, self.ma1a2max)))
        inside = tf.logical_and(inside, tf.logical_and(
            tf.greater(mb1b2, self.mb1b2min), tf.less(mb1b2, mb1b2max)))

        return inside

    @atfi.function
    def inside(self, x):

        inside = self.simple_inside(x)

        if self.mab1range or self.mab2range : 

          (pa1, pa2, pb1, pb2) = self.final_state_momenta(x)
          if self.mab1range : 
            mab1 = atfk.mass(pa1 + pa2 + pb1)
            inside = tf.logical_and(inside, tf.logical_and(
                       tf.greater(mab1, self.mab1range[0]), tf.less(mab1, self.mab1range[1])
                     ))
          if self.mab2range : 
            mab2 = atfk.mass(pa1 + pa2 + pb2)
            inside = tf.logical_and(inside, tf.logical_and(
                       tf.greater(mab2, self.mab2range[0]), tf.less(mab2, self.mab2range[1])
                     ))
        return inside

    @atfi.function
    def filter(self, x):
        y = tf.boolean_mask(x, self.simple_inside(x))
        return tf.boolean_mask(y, self.inside(y))

    @atfi.function
    def density(self, x):
        ma1a2 = self.m_a1a2(x)
        mb1b2 = self.m_b1b2(x)
        d1 = atfk.two_body_momentum(self.md, ma1a2, mb1b2)
        d2 = atfk.two_body_momentum(ma1a2, self.ma1, self.ma2)
        d3 = atfk.two_body_momentum(mb1b2, self.mb1, self.mb2)
        return d1*d2*d3/self.md

    @atfi.function
    def bounds(self):
        return [
            (self.ma1a2min, self.ma1a2max),
            (self.mb1b2min, self.mb1b2max),
            (self.costhamin, self.costhamax),
            (self.costhbmin, self.costhbmax),
            (-math.pi, math.pi)
        ]

    @atfi.function
    def unfiltered_sample(self, size, maximum=None):
        """
          Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [tf.random.uniform([size], self.ma1a2min, self.ma1a2max, dtype = atfi.fptype()),
             tf.random.uniform([size], self.mb1b2min, self.mb1b2max, dtype = atfi.fptype()),
             tf.random.uniform([size], self.costhamin, self.costhamax, dtype = atfi.fptype()),
             tf.random.uniform([size], self.costhbmin, self.costhbmax, dtype = atfi.fptype()),
             tf.random.uniform([size], -math.pi, math.pi, dtype = atfi.fptype()),
             ]
        if maximum is not None :
            v += [tf.random.uniform([size], 0., maximum, dtype = atfi.fptype())]
        return tf.stack(v, axis = 1)

    @atfi.function
    def uniform_sample(self, size, maximum=None):
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
    def m_a1a2(self, sample):
        """
          Return m2ab variable (vector) for the input sample
        """
        return sample[..., 0]

    @atfi.function
    def m_b1b2(self, sample):
        """
          Return m2bc variable (vector) for the input sample
        """
        return sample[..., 1]

    @atfi.function
    def cos_helicity_a(self, sample):
        """
          Return cos(helicity angle) of the A1A2 resonance
        """
        return sample[..., 2]

    @atfi.function
    def cos_helicity_b(self, sample):
        """
           Return cos(helicity angle) of the B1B2 resonance
        """
        return sample[..., 3]

    @atfi.function
    def phi(self, sample):
        """
           Return phi angle between A1A2 and B1B2 planes
        """
        return sample[..., 4]

    @atfi.function
    def final_state_momenta(self, x):
        """
           Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
           defined by the phase space vector x. The momenta are calculated in the
           D rest frame.
        """
        ma1a2 = self.m_a1a2(x)
        mb1b2 = self.m_b1b2(x)
        ctha = self.cos_helicity_a(x)
        cthb = self.cos_helicity_b(x)
        phi = self.phi(x)

        p0 = atfk.two_body_momentum(self.md, ma1a2, mb1b2)
        pA = atfk.two_body_momentum(ma1a2, self.ma1, self.ma2)
        pB = atfk.two_body_momentum(mb1b2, self.mb1, self.mb2)

        zeros = atfi.zeros(pA)

        p3A = atfk.rotate_euler(atfk.vector(zeros, zeros, pA), zeros, atfi.acos(ctha), zeros)
        p3B = atfk.rotate_euler(atfk.vector(zeros, zeros, pB), zeros, atfi.acos(cthb), phi)

        ea = atfi.sqrt(p0 ** 2 + ma1a2 ** 2)
        eb = atfi.sqrt(p0 ** 2 + mb1b2 ** 2)
        v0a = atfk.vector(zeros, zeros, p0 / ea)
        v0b = atfk.vector(zeros, zeros, -p0 / eb)

        p4A1 = atfk.lorentz_boost(atfk.lorentz_vector(p3A, atfi.sqrt(self.ma1 ** 2 + pA ** 2)), v0a)
        p4A2 = atfk.lorentz_boost(atfk.lorentz_vector(-p3A, atfi.sqrt(self.ma2 ** 2 + pA ** 2)), v0a)
        p4B1 = atfk.lorentz_boost(atfk.lorentz_vector(p3B, atfi.sqrt(self.mb1 ** 2 + pB ** 2)), v0b)
        p4B2 = atfk.lorentz_boost(atfk.lorentz_vector(-p3B, atfi.sqrt(self.mb2 ** 2 + pB ** 2)), v0b)

        return (p4A1, p4A2, p4B1, p4B2)

    def dimensionality(self):
        return 5
