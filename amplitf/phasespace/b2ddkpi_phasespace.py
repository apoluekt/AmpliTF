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
#import tensorflow as tf
import amplitf.interface as atfi
import amplitf.kinematics as atfk

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

class B2DDKpiPhaseSpace:

    def __init__(self, md, mk, mpi, mb, mddpirange=None):
        """
          Constructor
        """
        self.md = md
        self.mk = mk
        self.mpi = mpi
        self.mb = mb
        self.md2 = md*md
        self.mpi2 = mpi*mpi
        self.mb2 = mb*mb

        self.mddpirange = mddpirange

        self.smin = (2.*self.md+self.mpi)**2
        self.smax = (self.mb-self.mk)**2
        if mddpirange : 
          if self.smin < self.mddpirange[0]**2 : self.smin = self.mddpirange[0]**2
          if self.smax > self.mddpirange[1]**2 : self.smax = self.mddpirange[1]**2

        self.sigma1min = (self.md + self.mpi)**2
        self.sigma3min = (self.md + self.mpi)**2
        self.sigma1max = (math.sqrt(self.smax) - self.md)**2
        self.sigma3max = (math.sqrt(self.smax) - self.md)**2

    def inside(self, x):
        """
          Check if the point x is inside the phase space
        """
        s = self.s(x)
        sigma1 = self.sigma1(x)
        sigma3 = self.sigma3(x)
        costheta1 = self.costheta1(x)
        phi23 = self.phi23(x)

        inside = atfi.logical_and(atfi.greater(costheta1, -1.), atfi.less(costheta1, 1.))

        inside = atfi.logical_and(inside,
                                atfi.logical_and(atfi.greater(
                                    phi23, -math.pi), atfi.less(phi23, math.pi))
                                )

        mab = atfi.sqrt(sigma1)

        inside = atfi.logical_and(atfi.logical_and(atfi.greater(sigma1, self.sigma1min), atfi.less(sigma1, self.sigma1max)),
                                atfi.logical_and(atfi.greater(sigma3, self.sigma3min), atfi.less(sigma3, self.sigma3max)))

        inside = atfi.logical_and(inside, atfi.logical_and(atfi.greater(s, self.smin), atfi.less(s, self.smax)))

        # a: d
        # b: pi
        # c: d
        # m2ab: sigma1
        # m2bc: sigma2

        epi = (sigma1 - self.md2 + self.mpi2)/2./mab
        ed  = (s - sigma1 - self.md2)/2./mab
        p2pi = epi**2 - self.mpi2
        p2d = ed**2 - self.md2
        inside = atfi.logical_and(inside, atfi.logical_and(atfi.greater(p2pi, 0), atfi.greater(p2d, 0)))
        ppi = atfi.sqrt(p2pi)
        pd = atfi.sqrt(p2d)
        e2dpi = (ed+epi)**2
        sigma3_max = e2dpi - (pd - ppi)**2
        sigma3_min = e2dpi - (pd + ppi)**2

        return atfi.logical_and(inside, atfi.logical_and(atfi.greater(sigma3, sigma3_min), atfi.less(sigma3, sigma3_max)))

    def filter(self, x) :
        return x[self.inside(x)]

    def density(self, x) :
        s = self.s(x)
        return atfk.two_body_phase_space( self.mb, atfi.sqrt(s), self.mk)

    def bounds(self):
        return [
            (self.smin, self.smax),
            (self.sigma1min, self.sigma1max),
            (self.sigma3min, self.sigma3max),
            (-1., 1.),
            (-math.pi, math.pi)
        ]

    def unfiltered_sample(self, size, maximum=None):
        """
          Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [atfi.random_uniform([size], self.smin, self.smax),
             atfi.random_uniform([size], self.sigma1min, self.sigma1max),
             atfi.random_uniform([size], self.sigma3min, self.sigma3max),
             atfi.random_uniform([size], -1., 1.),
             atfi.random_uniform([size], -math.pi, math.pi),
             ]
        if maximum is not None :
            v += [tf.random_uniform([size], 0., maximum)]
        return tf.stack(v, axis = 1)

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

    def s(self, sample):
        return sample[..., 0]

    def sigma1(self, sample):
        return sample[..., 1]

    def sigma2(self, sample):
        s = self.s(sample)
        sigma1 = self.sigma1(sample)
        sigma3 = self.sigma3(sample)
        return s + 2.*self.md2 + self.mpi2 - sigma1 - sigma3

    def sigma3(self, sample):
        return sample[..., 2]

    def costheta1(self, sample):
        return sample[..., 3]

    def phi23(self, sample):
        return sample[..., 4]

    def dimensionality(self):
        return 5

    def final_state_momenta(self, x):
        s = self.s(x)
        sigma1 = self.sigma1(x)
        sigma3 = self.sigma3(x)
        costheta1 = self.costheta1(x)
        phi23 = self.phi23(x)

        m2ab = sigma1
        m2bc = sigma3
        m2ac = s + 2.*self.md2 + self.mpi2 - m2ab - m2bc
        sqrts = atfi.sqrt(s)

        p_a = atfk.two_body_momentum(sqrts, self.md,  atfi.sqrt(m2bc))
        p_b = atfk.two_body_momentum(sqrts, self.mpi, atfi.sqrt(m2ac))
        p_c = atfk.two_body_momentum(sqrts, self.md,  atfi.sqrt(m2ab))

        cos_theta_b = (p_a*p_a + p_b*p_b - p_c*p_c)/(2.*p_a*p_b)
        cos_theta_c = (p_a*p_a + p_c*p_c - p_b*p_b)/(2.*p_a*p_c)

        zeros = atfi.zeros(p_a)

        p4a = atfk.lorentz_vector(atfk.vector(zeros, zeros, p_a), atfi.sqrt(p_a ** 2 + self.md2))
        p4b = atfk.lorentz_vector(atfk.vector(p_b * atfi.sqrt(1. - cos_theta_b ** 2),
                                              zeros, -p_b * cos_theta_b), atfi.sqrt(p_b ** 2 + self.mpi2))
        p4c = atfk.lorentz_vector(atfk.vector(-p_c * atfi.sqrt(1. - cos_theta_c ** 2),
                                              zeros, -p_c * cos_theta_c), atfi.sqrt(p_c ** 2 + self.md2))

        p_x = atfk.two_body_momentum(self.mb, self.mk, sqrts)

        p4d1 = atfk.rotate_lorentz_vector(p4a, phi23, atfi.acos(costheta1), zeros )
        p4pi = atfk.rotate_lorentz_vector(p4b, phi23, atfi.acos(costheta1), zeros )
        p4d2 = atfk.rotate_lorentz_vector(p4c, phi23, atfi.acos(costheta1), zeros )

        p4k = atfk.lorentz_vector(atfk.vector( zeros, zeros, -p_x), atfi.sqrt(p_x ** 2 + self.mk**2))
        p4x = atfk.lorentz_vector(atfk.vector( zeros, zeros,  p_x), atfi.sqrt(p_x ** 2 + s))

        p4d1 = atfk.boost_from_rest(p4d1, p4x)
        p4pi = atfk.boost_from_rest(p4pi, p4x)
        p4d2 = atfk.boost_from_rest(p4d2, p4x)

        return (p4d1, p4d2, p4k, p4pi)
