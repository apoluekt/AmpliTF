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
import amplitf.interface as atfi
import amplitf.kinematics as atfk
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class Baryonic3BodyPhaseSpace(DalitzPhaseSpace):
    """
      Derived class for baryonic 3-body decay, baryon -> scalar scalar baryon
      Include 2D phase-space + 3 decay plane orientation angular variables, for full polarization treatment
    """

    def cos_theta_a(self, sample):
        """
          Return thetaa variable (vector) for the input sample
        """
        return sample[..., 2]

    def phi_a(self, sample):
        """
          Return phia variable (vector) for the input sample
        """
        return sample[..., 3]

    def phi_bc(self, sample):
        """
          Return phibc variable (vector) for the input sample
        """
        return sample[..., 4]

    def inside(self, x):
        """
          Check if the point x=(m2ab, m2bc, cos_theta_a, phi_a, phi_bc) is inside the phase space
        """
        m2ab = self.m2ab(x)
        m2bc = self.m2bc(x)
        mab = atfi.sqrt(m2ab)
        costhetaa = self.cos_theta_a(x)
        phia = self.phi_a(x)
        phibc = self.phi_bc(x)

        inside = atfi.logical_and(atfi.logical_and(atfi.greater(m2ab, self.minab), atfi.less(m2ab, self.maxab)),
                                  atfi.logical_and(atfi.greater(m2bc, self.minbc), atfi.less(m2bc, self.maxbc)))

        if self.macrange:
            m2ac = self.msqsum - m2ab - m2bc
            inside = atfi.logical_and(inside, atfi.logical_and(atfi.greater(
                m2ac, self.macrange[0]**2), atfi.less(m2ac, self.macrange[1]**2)))

        if self.symmetric:
            inside = atfi.logical_and(inside, atfi.greater(m2bc, m2ab))

        eb = (m2ab - self.ma2 + self.mb2)/2./mab
        ec = (self.md2 - m2ab - self.mc2)/2./mab
        p2b = eb**2 - self.mb2
        p2c = ec**2 - self.mc2
        inside = atfi.logical_and(inside, atfi.logical_and(
            atfi.greater(p2c, 0), atfi.greater(p2b, 0)))

        pb = atfi.sqrt(p2b)
        pc = atfi.sqrt(p2c)
        e2bc = (eb+ec)**2
        m2bc_max = e2bc - (pb - pc)**2
        m2bc_min = e2bc - (pb + pc)**2

        inside_phsp = atfi.logical_and(inside, atfi.logical_and(
            atfi.greater(m2bc, m2bc_min), atfi.less(m2bc, m2bc_max)))

        inside_theta = atfi.logical_and(atfi.greater(
            costhetaa, -1.), atfi.less(costhetaa, 1.))
        inside_phi = atfi.logical_and(atfi.logical_and(atfi.greater(phia, -1.*math.pi), atfi.less(phia, math.pi)),
                                      atfi.logical_and(atfi.greater(phibc, -1.*math.pi), atfi.less(phibc, math.pi)))
        inside_ang = atfi.logical_and(inside_theta, inside_phi)

        return atfi.logical_and(inside_phsp, inside_ang)

    def filter(self, x):
        return x[self.inside(x)]

    def unfiltered_sample(self, size, maximum = None):
        """
          Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [atfi.random_uniform([size], self.minab, self.maxab),
             atfi.random_uniform([size], self.minbc, self.maxbc),
             atfi.random_uniform([size], -1., 1.),
             atfi.random_uniform([size], -1. * math.pi, math.pi),
             atfi.random_uniform([size], -1. * math.pi, math.pi)
            ]

        if maximum is not None :
            v += [atfi.random_uniform([size], 0., maximum)]
        return atfi.stack(v, axis = 1)

    def final_state_momenta(self, m2ab, m2bc, costhetaa, phia, phibc):
        """
          Calculate 4-momenta of final state tracks in the 5D phase space
            m2ab, m2bc : invariant masses of AB and BC combinations
            (cos)thetaa, phia : direction angles of the particle A in the D reference frame
            phibc : angle of BC plane wrt. polarisation plane z x p_a
        """

        thetaa = atfi.acos(costhetaa)

        m2ac = self.msqsum - m2ab - m2bc

        # Magnitude of the momenta
        p_a = atfk.two_body_momentum(self.md, self.ma, atfi.sqrt(m2bc))
        p_b = atfk.two_body_momentum(self.md, self.mb, atfi.sqrt(m2ac))
        p_c = atfk.two_body_momentum(self.md, self.mc, atfi.sqrt(m2ab))

        cos_theta_b = (p_a*p_a + p_b*p_b - p_c*p_c)/(2.*p_a*p_b)
        cos_theta_c = (p_a*p_a + p_c*p_c - p_b*p_b)/(2.*p_a*p_c)

        # Fix momenta with p3a oriented in z (quantisation axis) direction
        p3a = atfk.vector(atfi.zeros(p_a), atfi.zeros(p_a), p_a)
        p3b = atfk.vector(p_b * Sqrt(1. - cos_theta_b ** 2),
                          atfi.zeros(p_b), -p_b * cos_theta_b)
        p3c = atfk.vector(-p_c * Sqrt(1. - cos_theta_c ** 2),
                          atfi.zeros(p_c), -p_c * cos_theta_c)

        # rotate vectors to have p3a with thetaa as polar helicity angle
        p3a = atfk.rotate_euler(p3a, atfi.const(0.), thetaa, atfi.const(0.))
        p3b = atfk.rotate_euler(p3b, atfi.const(0.), thetaa, atfi.const(0.))
        p3c = atfk.rotate_euler(p3c, atfi.const(0.), thetaa, atfi.const(0.))

        # rotate vectors to have p3a with phia as azimuthal helicity angle
        p3a = atfk.rotate_euler(p3a, phia, atfi.const(0.), atfi.const(0.))
        p3b = atfk.rotate_euler(p3b, phia, atfi.const(0.), atfi.const(0.))
        p3c = atfk.rotate_euler(p3c, phia, atfi.const(0.), atfi.const(0.))

        # rotate BC plane to have phibc as angle with the polarization plane
        p3b = atfk.rotate(p3b, phibc, p3a)
        p3c = atfk.rotate(p3c, phibc, p3a)

        # Define 4-vectors
        p4a = atfk.lorentz_vector(p3a, atfi.sqrt(p_a ** 2 + self.ma2))
        p4b = atfk.lorentz_vector(p3b, atfi.sqrt(p_b ** 2 + self.mb2))
        p4c = atfk.lorentz_vector(p3c, atfi.sqrt(p_c ** 2 + self.mc2))

        return (p4a, p4b, p4c)

    def dimensionality(self):
        return 5
