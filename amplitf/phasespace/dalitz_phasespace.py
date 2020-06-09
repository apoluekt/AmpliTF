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

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

class DalitzPhaseSpace:
    """
    Class for Dalitz plot (2D) phase space for the 3-body decay D->ABC
    """

    def __init__(self, ma, mb, mc, md, mabrange=None, mbcrange=None, macrange=None, symmetric=False):
        """
        Constructor
          ma - A mass
          mb - B mass
          mc - C mass
          md - D (mother) mass
        """
        self.ma = ma
        self.mb = mb
        self.mc = mc
        self.md = md
        self.ma2 = ma*ma
        self.mb2 = mb*mb
        self.mc2 = mc*mc
        self.md2 = md*md
        self.msqsum = self.md2 + self.ma2 + self.mb2 + self.mc2
        self.minab = (ma + mb)**2
        self.maxab = (md - mc)**2
        self.minbc = (mb + mc)**2
        self.maxbc = (md - ma)**2
        self.minac = (ma + mc)**2
        self.maxac = (md - mb)**2
        self.macrange = macrange
        self.symmetric = symmetric
        self.min_mprimeac = 0.0
        self.max_mprimeac = 1.0
        self.min_thprimeac = 0.0
        self.max_thprimeac = 1.0
        if self.symmetric:
            self.max_thprimeac = 0.5
        if mabrange:
            if mabrange[1]**2 < self.maxab:
                self.maxab = mabrange[1]**2
            if mabrange[0]**2 > self.minab:
                self.minab = mabrange[0]**2
        if mbcrange:
            if mbcrange[1]**2 < self.maxbc:
                self.maxbc = mbcrange[1]**2
            if mbcrange[0]**2 > self.minbc:
                self.minbc = mbcrange[0]**2

    def inside(self, x):
        """
          Check if the point x=(m2ab, m2bc) is inside the phase space
        """
        m2ab = self.m2ab(x)
        m2bc = self.m2bc(x)
        mab = atfi.sqrt(m2ab)

        inside = atfi.logical_and(atfi.logical_and(atfi.greater(m2ab, self.minab), atfi.less(m2ab, self.maxab)),
                                  atfi.logical_and(atfi.greater(m2bc, self.minbc), atfi.less(m2bc, self.maxbc)))

        if self.macrange:
            m2ac = self.msqsum - m2ab - m2bc
            inside = atfi.logical_and(inside, atfi.logical_and(atfi.greater(m2ac, self.macrange[0]**2), atfi.less(m2ac, self.macrange[1]**2)))

        if self.symmetric:
            inside = atfi.logical_and(inside, atfi.greater(m2bc, m2ab))

        eb = (m2ab - self.ma2 + self.mb2)/2./mab
        ec = (self.md2 - m2ab - self.mc2)/2./mab
        p2b = eb**2 - self.mb2
        p2c = ec**2 - self.mc2
        inside = atfi.logical_and(inside, atfi.logical_and(atfi.greater(p2c, 0), atfi.greater(p2b, 0)))
        pb = atfi.sqrt(p2b)
        pc = atfi.sqrt(p2c)
        e2bc = (eb+ec)**2
        m2bc_max = e2bc - (pb - pc)**2
        m2bc_min = e2bc - (pb + pc)**2
        return atfi.logical_and(inside, atfi.logical_and(atfi.greater(m2bc, m2bc_min), atfi.less(m2bc, m2bc_max)))

    def filter(self, x):
        return x[self.inside(x)]

    def unfiltered_sample(self, size, maximum = None):
        """
          Return TF graph for uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [atfi.random_uniform([size], self.minab, self.maxab),
             atfi.random_uniform([size], self.minbc, self.maxbc) ]

        if maximum is not None :
            v += [ atfi.random_uniform([size], 0., maximum) ]
        return atfi.stack(v, axis = 1)

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

    def rectangular_grid_sample(self, size_ab, size_bc, space_to_sample="DP"):
        """
          Create a data sample in the form of rectangular grid of points within the phase space.
          Useful for normalisation.
            size_ab : number of grid nodes in m2ab range
            size_bc : number of grid nodes in m2bc range
            space_to_sample: Sampling is done according to cases below but all of them return DP vars (m^2_{ab}, m^2_{bc}).
                -if 'DP': Unifrom sampling is in (m^2_{ab}, m^2_{bc})
                -if 'linDP': Samples in (m_{ab}, m_{bc})
                -if 'sqDP': Samples in (mPrimeAC, thPrimeAC).
        """
        size = size_ab * size_bc
        mgrid = np.lib.index_tricks.nd_grid()
        if space_to_sample == "linDP":
            vab = (mgrid[0:size_ab, 0:size_bc][0] * (math.sqrt(self.maxab) -
                                                     math.sqrt(self.minab)) / float(size_ab) + math.sqrt(self.minab)) ** 2.
            vbc = (mgrid[0:size_ab, 0:size_bc][1] * (math.sqrt(self.maxbc) -
                                                     math.sqrt(self.minbc)) / float(size_bc) + math.sqrt(self.minbc)) ** 2.
            v = [vab.reshape(size).astype('d'), vbc.reshape(size).astype('d')]
            dlz = atfi.stack(v, axis=1)
        elif space_to_sample == "sqDP":
            x = np.linspace(self.min_mprimeac, self.max_mprimeac, size_ab)
            y = np.linspace(self.min_thprimeac, self.max_thprimeac, size_bc)
            # Remove corners of sqDP as they lie outside phsp
            xnew = x[(x > self.min_mprimeac) & (x < self.max_mprimeac) & (
                y > self.min_thprimeac) & (y < self.max_thprimeac)]
            ynew = y[(x > self.min_mprimeac) & (x < self.max_mprimeac) & (
                y > self.min_thprimeac) & (y < self.max_thprimeac)]
            mprimeac, thprimeac = np.meshgrid(xnew, ynew)
            dlz = self.from_square_dalitz_plot(
                mprimeac.flatten().astype('d'), thprimeac.flatten().astype('d'))
        else:
            vab = mgrid[0:size_ab, 0:size_bc][0] * \
                  (self.maxab-self.minab) / float(size_ab) + self.minab
            vbc = mgrid[0:size_ab, 0:size_bc][1] * \
                  (self.maxbc-self.minbc) / float(size_bc) + self.minbc
            v = [vab.reshape(size).astype('d'), vbc.reshape(size).astype('d')]
            dlz = atfi.stack(v, axis=1)

        return self.filter(dlz)
 
    def m2ab(self, sample):
        """
          Return m2ab variable (vector) for the input sample
        """
        return sample[..., 0]

    def m2bc(self, sample):
        """
           Return m2bc variable (vector) for the input sample
        """
        return sample[..., 1]

    def m2ac(self, sample):
        """
          Return m2ac variable (vector) for the input sample.
          It is calculated from m2ab and m2bc
        """
        return self.msqsum - self.m2ab(sample) - self.m2bc(sample)

    def cos_helicity_ab(self, sample):
        """
          Calculate cos(helicity angle) of the AB resonance
        """
        return atfk.cos_helicity_angle_dalitz(self.m2ab(sample), self.m2bc(sample), self.md, self.ma, self.mb, self.mc)

    def cos_helicity_bc(self, sample):
        """
           Calculate cos(helicity angle) of the BC resonance
        """
        return atfk.cos_helicity_angle_dalitz(self.m2bc(sample), self.m2ac(sample), self.md, self.mb, self.mc, self.ma)

    def cos_helicity_ac(self, sample):
        """
           Calculate cos(helicity angle) of the AC resonance
        """
        return atfk.cos_helicity_angle_dalitz(self.m2ac(sample), self.m2ab(sample), self.md, self.mc, self.ma, self.mb)

    def m_prime_ac(self, sample):
        """
          Square Dalitz plot variable m'
        """
        mac = atfi.sqrt(self.m2ac(sample))
        return atfi.acos(2 * (mac - math.sqrt(self.minac)) / (math.sqrt(self.maxac) - math.sqrt(self.minac)) - 1.) / math.pi

    def theta_prime_ac(self, sample):
        """
          Square Dalitz plot variable theta'
        """
        return atfi.acos(self.cos_helicity_ac(sample)) / math.pi

    def m_prime_ab(self, sample):
        """
          Square Dalitz plot variable m'
        """
        mab = atfi.sqrt(self.m2ab(sample))
        return atfi.acos(2 * (mab - math.sqrt(self.minab)) / (math.sqrt(self.maxab) - math.sqrt(self.minab)) - 1.) / math.pi

    def from_square_dalitz_plot(self, mprimeac, thprimeac):
        """
          sample: Given mprimeac and thprimeac, returns 2D tensor for (m2ab, m2bc). 
          Make sure you don't pass in sqDP corner points as they lie outside phsp.
        """
        m2AC = 0.25*(self.maxac ** 0.5 * atfi.cos(math.pi * mprimeac) + self.maxac **
                     0.5 - self.minac ** 0.5 * atfi.cos(math.pi*mprimeac) + self.minac ** 0.5)**2
        m2AB = 0.5*(-m2AC**2 + m2AC * self.ma ** 2 + m2AC * self.mb ** 2 + m2AC * self.mc ** 2 + m2AC * self.md ** 2 -
                    m2AC * atfi.sqrt((m2AC * (m2AC - 2.0 * self.ma ** 2 - 2.0 * self.mc ** 2) +
                    self.ma ** 4 - 2.0 * self.ma ** 2 * self.mc ** 2 + self.mc ** 4) / m2AC) * atfi.sqrt((m2AC * (m2AC - 2.0 * self.mb ** 2 - 2.0 * self.md ** 2) +
                    self.mb ** 4 - 2.0 * self.mb ** 2 * self.md ** 2 + self.md ** 4) / m2AC) * atfi.cos(math.pi * thprimeac) -
                    self.ma ** 2 * self.mb ** 2 + self.ma ** 2 * self.md ** 2 + self.mb ** 2 * self.mc ** 2 - self.mc ** 2 * self.md ** 2)/m2AC
        m2BC = self.msqsum - m2AC - m2AB
        return atfi.stack([m2AB, m2BC], axis=1)

    def square_dalitz_plot_jacobian(self, sample):
        """
          sample: [mAB^2, mBC^2]
          Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dMpr*dThpr where Mpr, Thpr are defined in (AC) frame.
        """
        mPrime = self.m_prime_ac(sample)
        thPrime = self.theta_prime_ac(sample)

        diff_AC = atfi.cast_real(atfi.sqrt(self.maxac) - atfi.sqrt(self.minac))
        mAC = atfi.const(0.5) * diff_AC * (Const(1.) + atfi.cos(atfi.pi() * mPrime)
                                           ) + atfi.cast_real(atfi.sqrt(self.minac))
        mACSq = mAC*mAC

        eAcmsAC = atfi.const(0.5) * (mACSq - atfi.cast_real(self.mc2) + atfi.cast_real(self.ma2)) / mAC
        eBcmsAC = atfi.const(0.5) * (atfi.cast_real(self.md, atfi.fptype()) ** 2. - mACSq - atfi.cast_real(self.mb2)) / mAC

        pAcmsAC = atfi.sqrt(eAcmsAC ** 2. - atfi.cast_real(self.ma2))
        pBcmsAC = atfi.sqrt(eBcmsAC ** 2. - atfi.cast_real(self.mb2))

        deriv1 = Pi() * atfi.const(0.5) * diff_AC * atfi.sin(atfi.pi() * mPrime)
        deriv2 = Pi() * atfi.sin(atfi.pi() * thPrime)

        return atfi.const(4.) * pAcmsAC * pBcmsAC * mAC * deriv1 * deriv2

    def invariant_mass_jacobian(self, sample):
        """
          sample: [mAB^2, mBC^2]
          Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dmAB*dmBC. |J| = 4*mAB*mBC
        """
        return atfi.const(4.) * atfi.sqrt(self.m2ab(sample)) * atfi.sqrt(self.m2bc(sample))

    def theta_prime_ab(self, sample):
        """
          Square Dalitz plot variable theta'
        """
        return atfi.acos(-self.cos_helicity_ab(sample)) / math.pi

    def m_prime_bc(self, sample):
        """
          Square Dalitz plot variable m'
        """
        mbc = atfi.sqrt(self.m2bc(sample))
        return atfi.acos(2 * (mbc - math.sqrt(self.minbc)) / (math.sqrt(self.maxbc) - math.sqrt(self.minbc)) - 1.) / math.pi

    def theta_prime_bc(self, sample):
        """
          Square Dalitz plot variable theta'
        """
        return atfi.acos(-self.cos_helicity_bc(sample)) / math.pi

    def from_vectors(self, m2ab, m2bc):
        """
          Create Dalitz plot tensor from two vectors of variables, m2ab and m2bc
        """
        return atfi.stack([m2ab, m2bc], axis=1)

    def final_state_momenta(self, m2ab, m2bc):
        """
          Calculate 4-momenta of final state tracks in a certain reference frame
          (decay is in x-z plane, particle A moves along z axis)
            m2ab, m2bc : invariant masses of AB and BC combinations
        """

        m2ac = self.msqsum - m2ab - m2bc

        p_a = atfk.two_body_momentum(self.md, self.ma, atfi.sqrt(m2bc))
        p_b = atfk.two_body_momentum(self.md, self.mb, atfi.sqrt(m2ac))
        p_c = atfk.two_body_momentum(self.md, self.mc, atfi.sqrt(m2ab))

        cos_theta_b = (p_a*p_a + p_b*p_b - p_c*p_c)/(2.*p_a*p_b)
        cos_theta_c = (p_a*p_a + p_c*p_c - p_b*p_b)/(2.*p_a*p_c)

        p4a = atfk.lorentz_vector(atfk.vector(atfi.zeros(p_a), atfi.zeros(p_a),
                                              p_a), atfi.sqrt(p_a ** 2 + self.ma2))
        p4b = atfk.lorentz_vector(atfk.vector(p_b * atfi.sqrt(1. - cos_theta_b ** 2),
                                              atfi.zeros(p_b), -p_b * cos_theta_b), atfi.sqrt(p_b ** 2 + self.mb2))
        p4c = atfk.lorentz_vector(atfk.vector(-p_c * atfi.sqrt(1. - cos_theta_c ** 2),
                                              atfi.zeros(p_c), -p_c * cos_theta_c), atfi.sqrt(p_c ** 2 + self.mc2))
        return (p4a, p4b, p4c)

    def dimensionality(self):
        return 2
