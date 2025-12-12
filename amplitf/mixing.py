# Copyright 2024 CERN for the benefit of the LHCb collaboration
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

r"""
Formulas for building a model where meson mixing is involved.

The mixing parameters for the mass eigenstates of a neutral meson (:math:`M^0`, :math:`\bar{M}^0`) are

.. math::

    \begin{align}
    x &\equiv 2\frac{m_2-m_1}{\gamma_1+\gamma_2} = \frac{\Delta M}{\Gamma}\\
    y &\equiv \frac{\gamma_2-\gamma_1}{\gamma_1+\gamma_2} = \frac{\Delta\Gamma}{2\Gamma},
    \end{align}


where :math:`m_i` and :math:`\gamma_i` (:math:`i=1,2`) are the eigenvalues of the phyical eigenstates :math:`M_{1,2}`.

The time evolution of the physical states is given by

.. math::
    
    |M_i(t)>= e^{-i(m_i-i\gamma_i/2)t}|M_i(t=0)>.


... Continue explanation later
"""
import tensorflow as tf
import amplitf.interface as atfi
import amplitf.dynamics as atfd


# Time evolution functions.
@atfi.function
def psip( t, y, tau ):
    r"""Time evolution function :math:`\psi_+(t)`

    Args:
        t (float): decay time of the candidate
        y (float): mixing parameter
        tau (float): lifetime of the decaying particle

    Returns:
        float: the time evolution function for the sum of the two decay amplitudes
    """
    return atfi.exp( - ( 1.0 + y ) * t / tau)


@atfi.function
def psim( t, y, tau ):
    r"""Time evolution function :math:`\psi_-(t)`

    Args:
        t (float): decay time of the candidate
        x (float): mixing parameter
        tau (float): lifetime of the decaying particle

    Returns:
        float: the time evolution function for the sum of the two decay amplitudes
    """
    return atfi.exp( - ( 1.0 - y ) * t / tau)


@atfi.function
def psii( t, x, tau ):
    r"""Time evolution function :math:`\psi_i(t)`

    Args:
        t (float): decay time of the candidate
        x (float): mixing parameter
        tau (float): lifetime of the decaying particle

    Returns:
        float: the time evolution function for the sum of the two decay amplitudes
    """
    return atfi.exp( - atfi.complex( atfi.const(1.0) , x ) * tf.cast(t / tau, tf.complex128))


# Probability density
def mixing_density(ampl_dir, ampl_cnj, qoverp,
                   time_evolution_pos, time_evolution_neg, time_evolution_int):
    """Calculates the probability density of the mixing

    Args:
        ampl_dir (complex): amplitude of the direct decay
        ampl_cnj (complex): amplitude of the charged conjugate decay
        qoverp (complex): CP violation
        time_evolution_pos (float): time evolution of the sum of the decays
        time_evolution_neg (float): time evolution of the subtraction of the decays
        time_evolution_int (complex): time evolution of the interference of the decays

    Returns:
        float: the density of probability for the event
    """    
    ampDir = ampl_dir
    ampCnj = qoverp * ampl_cnj

    apb2 = 0.5 * (ampDir + ampCnj)
    amb2 = 0.5 * atfi.conjugate(ampDir - ampCnj)

    dens  = atfd.density( apb2 ) * time_evolution_pos
    dens += atfd.density( amb2 ) * time_evolution_neg
    dens += 2.0 * tf.math.real( apb2 * amb2 * time_evolution_int )
    return dens
