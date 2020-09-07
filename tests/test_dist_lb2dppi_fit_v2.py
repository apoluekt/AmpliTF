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

#
# Example of three-body amplitude fit for baryonic decay Lambda_b^0 -> D0 p pi-
# The initial Lambda_b0 is assumed to be unpolarised, so only two degrees of 
# freedom remain, the fit is two-dimensional. 
#
# The amplitude formalism is taken from https://arxiv.org/abs/1701.07873
#

import tensorflow as tf

import sys, os, math
sys.path.append("../")
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import argparse
import amplitf.interface as atfi
import numpy as np
import matplotlib.pyplot as plt

atfi.backend_auto()

import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.toymc as atft
import amplitf.likelihood as atfl
import amplitf.optimisation as atfo
import amplitf.rootio as atfr
import amplitf.plotting as atfp
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from iminuit import Minuit
from timeit import default_timer as timer

# Calculate orbital momentum for a decay of a particle 
# of a given spin and parity to a proton (J^p=1/2+) and a pseudoscalar.
# All the spins and total momenta are expressed in units of 1/2
def OrbitalMomentum(spin, parity) : 
  l1 = (spin-1)/2     # Lowest possible momentum
  p1 = 2*(l1 % 2)-1   # p=(-1)^(L1+1), e.g. p=-1 if L=0
  if p1 == parity : return l1
  return l1+1

# Return the sign in front of the complex coupling
# for amplitudes with baryonic intermediate resonances
# See Eq. (3), page 3 of LHCB-ANA-2015-072, 
# https://svnweb.cern.ch/cern/wsvn/lhcbdocs/Notes/ANA/2015/072/drafts/lb2dppi_aman_v3r4.pdf
def CouplingSign(spin, parity) : 
  jp =  1
  jd =  0
  pp =  1
  pd = -1
  s = 2*(((jp+jd-spin)/2+1) % 2)-1
  s *= (pp*pd*parity)
  return s

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description = 'test tf.distribute')
    parser.add_argument('--strategy', type=str, nargs='?', default='onedevice')
    parser.add_argument('--nograd', action='store_true')
    argsp = parser.parse_args()
    
    if argsp.strategy == 'onedevice':
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif argsp.strategy == 'mirrored' :
        strategy = tf.distribute.MirroredStrategy()
    elif argsp.strategy == 'centralstorage' :
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    else :
        print('{} strategy not available. Choose `onedevice`, `mirrored` or `centralstorage`'.format(argsp.strategy))
        exit()
    
    use_gradient = not argsp.nograd
    
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Default flags that can be overridden by command line options
    norm_grid = 400
    toy_sample = 60000

    # Masses of initial and final state particles
    mlb = 5.620
    md  = 1.865
    mpi = 0.140
    mp  = 0.938

    # Create phase space object for 3-body baryonic decay
    # Use only a subrange of D0p invariant masses
    phsp = DalitzPhaseSpace(md, mp, mpi, mlb, mabrange = (0., 3.) )

    # Constant parameters of intermediate resonances
    mass_lcst   = atfi.const(2.88153)
    width_lcst  = atfi.const(0.0058)

    mass_lcx    = atfi.const(2.857)
    width_lcx   = atfi.const(0.060)

    mass_lcstst   = atfi.const(2.945)
    width_lcstst  = atfi.const(0.026)

    mass0 = atfi.const(3.)

    # Blatt-Weisskopf radii
    db = atfi.const(5.)
    dr = atfi.const(1.5)

    pars = [

      # Slope parameters for exponential nonresonant amplitudes
      atfi.FitParameter("alpha12p", 2.3, 0., 10., 0.01),
      atfi.FitParameter("alpha12m", 1.0, 0., 10., 0.01),
      atfi.FitParameter("alpha32p", 2.5, 0., 10., 0.01),
      atfi.FitParameter("alpha32m", 2.6, 0., 10., 0.01),

      # List of complex couplings
      atfi.FitParameter("ArX1", -0.38, -10., 10., 0.01), atfi.FitParameter("AiX1",  0.86, -10., 10., 0.01), 
      atfi.FitParameter("ArX2",  6.59, -10., 10., 0.01), atfi.FitParameter("AiX2", -0.38, -10., 10., 0.01),
      atfi.FitParameter("Ar29401",  0.53, -10., 10., 0.01), atfi.FitParameter("Ai29401", 0.14, -10., 10., 0.01), 
      atfi.FitParameter("Ar29402", -1.24, -10., 10., 0.01), atfi.FitParameter("Ai29402", 0.02, -10., 10., 0.01),
      atfi.FitParameter("Ar12p1",  0.05, -10., 10., 0.01), atfi.FitParameter("Ai12p1",  0.23, -10., 10., 0.01), 
      atfi.FitParameter("Ar12p2", -0.16, -10., 10., 0.01), atfi.FitParameter("Ai12p2", -2.86, -10., 10., 0.01),
      atfi.FitParameter("Ar12m1",  1.17, -10., 10., 0.01), atfi.FitParameter("Ai12m1", 0.76, -10., 10., 0.01), 
      atfi.FitParameter("Ar12m2", -2.55, -10., 10., 0.01), atfi.FitParameter("Ai12m2", 3.86, -10., 10., 0.01),
      atfi.FitParameter("Ar32p1",  0., -100., 100., 0.01), atfi.FitParameter("Ai32p1",  0., -100., 100., 0.01), 
      atfi.FitParameter("Ar32p2",  0., -100., 100., 0.01), atfi.FitParameter("Ai32p2",  0., -100., 100., 0.01),
      atfi.FitParameter("Ar32m1",  0.95, -10., 10., 0.01), atfi.FitParameter("Ai32m1", -0.45, -10., 10., 0.01), 
      atfi.FitParameter("Ar32m2", -2.27, -10., 10., 0.01), atfi.FitParameter("Ai32m2",  0.95, -10., 10., 0.01),
    ]

    res_names = ["X", "2940", "12p", "12m", "32p", "32m" ]

    # Model description
    #@atfi.function
    def model(x, params, switches = 7*[ 1 ]) : 

      m2dp  = phsp.m2ab(x)
      m2ppi = phsp.m2bc(x)

      p4d, p4p, p4pi = phsp.final_state_momenta(m2dp, m2ppi)
      dp_theta_r, dp_phi_r, dp_theta_d, dp_phi_d = atfk.helicity_angles_3body(p4d, p4p, p4pi)

      alpha12p = params["alpha12p"]
      alpha12m = params["alpha12m"]
      alpha32p = params["alpha32p"]
      alpha32m = params["alpha32m"]

      couplings = [ ((params[f"Ar{n}1"], params[f"Ai{n}1"]), (params[f"Ar{n}2"], params[f"Ai{n}2"])) for n in res_names ]

      # List of intermediate resonances corresponds to arXiv link above. 
      resonances = [
      (atfd.breit_wigner_lineshape(m2dp, mass_lcst, width_lcst, md, mp, mpi, mlb, dr, db, OrbitalMomentum(5, 1), 2), 5, 1,
       (atfi.const(1.), atfi.const(0.)), (atfi.const(0.), atfi.const(0.))
      ), 
      (atfd.breit_wigner_lineshape(m2dp, mass_lcx, width_lcx, md, mp, mpi, mlb, dr, db, OrbitalMomentum(3, 1), 1), 3, 1,
       couplings[0][0], couplings[0][1]
      ), 
      (atfd.breit_wigner_lineshape(m2dp, mass_lcstst, width_lcstst, md, mp, mpi, mlb, dr, db, OrbitalMomentum(3, -1), 1), 3, -1,
       couplings[1][0], couplings[1][1]
      ), 
      (atfd.exponential_nonresonant_lineshape(m2dp, mass0, alpha12p, md, mp, mpi, mlb, OrbitalMomentum(1, 1), 0), 1, 1,
       couplings[2][0], couplings[2][1]
      ), 
      (atfd.exponential_nonresonant_lineshape(m2dp, mass0, alpha12m, md, mp, mpi, mlb, OrbitalMomentum(1, -1), 0), 1, -1,
       couplings[3][0], couplings[3][1]
      ),
      (atfd.exponential_nonresonant_lineshape(m2dp, mass0, alpha32p, md, mp, mpi, mlb, OrbitalMomentum(3, 1), 1), 3, 1,
       couplings[4][0], couplings[4][1]
      ), 
      (atfd.exponential_nonresonant_lineshape(m2dp, mass0, alpha32m, md, mp, mpi, mlb, OrbitalMomentum(3, -1), 1), 3, -1,
       couplings[5][0], couplings[5][1]
      ), 
      ]

      #density = atfi.const(0.)
      density = 0.

      # Decay density is an incoherent sum over initial and final state polarisations 
      # (assumong no polarisation for Lambda_b^0), and for each polarisation combination 
      # it is a coherent sum over intermediate states (including two polarisations of 
      # the intermediate resonance). 
      for pol_lb in [-1, 1] : 
        for pol_p in [-1, 1] : 
          ampl = atfi.complex(atfi.const(0.), atfi.const(0.))
          for i,r in enumerate(resonances) : 
            lineshape = r[0]
            spin = r[1]
            parity = r[2]
            if pol_p == -1 : 
              sign = CouplingSign(spin, parity)
              coupling1 = atfi.complex(r[3][0], r[3][1]) * sign
              coupling2 = atfi.complex(r[4][0], r[4][1]) * sign
            else : 
              coupling1 = atfi.complex(r[3][0], r[3][1])
              coupling2 = atfi.complex(r[4][0], r[4][1])
            if switches[i] : 
              ampl += coupling1*lineshape*\
                  atfk.helicity_amplitude_3body(dp_theta_r, dp_phi_r, dp_theta_d, dp_phi_d, 1, spin, pol_lb, 1, 0, pol_p, 0)
              ampl += coupling2*lineshape*\
                  atfk.helicity_amplitude_3body(dp_theta_r, dp_phi_r, dp_theta_d, dp_phi_d, 1, spin, pol_lb, -1, 0, pol_p, 0)
          density += atfd.density(ampl)

      return density

    # Load data and norm samples
    norm_sample = np.load('test_norm.npy', allow_pickle=True)
    data_sample = np.load('test_data.npy', allow_pickle=True)

    global_batch_size = data_sample.shape[0]
    tf_dataset = tf.data.Dataset.from_tensor_slices(data_sample).batch(global_batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(tf_dataset)
    dist_values = iter(dist_dataset).get_next()

    tf_dataset_norm = tf.data.Dataset.from_tensor_slices(norm_sample).batch(global_batch_size)
    dist_dataset_norm = strategy.experimental_distribute_dataset(tf_dataset_norm)
    dist_values_norm = iter(dist_dataset_norm).get_next()

    @atfi.function
    def nll(data, norm, params) :
        def integral(norm, params):
            return atfl.integral(model(norm, params))
        integral_per_replica = strategy.run(integral, args=(norm, params))
        integral = strategy.reduce('MEAN', integral_per_replica, axis=None)
        def loss(data, params):
            return atfl.unbinned_nll(model(data, params), integral)
        per_replica = strategy.run(loss, args=(data, params))
        return strategy.reduce('SUM', per_replica, axis=None)

    float_pars = [p for p in pars if p.floating() ]
    init_pars = [p.init_value for p in pars if p.floating()]
    kwargs = { p.name : p() for p in float_pars }
    print("nll ", nll(dist_values, dist_values_norm, kwargs)) #First call
    atfi.gradient.n = 0
    gradient_supported = atfi.create_gradient(nll, (dist_values, dist_values_norm), float_pars)
    print("gradient ", atfi.gradient(init_pars)) #First call gradient
    
    result = atfo.run_minuit(nll, pars, args = (dist_values, dist_values_norm), use_gradient = use_gradient)
    print(result)

    wmode = 'a' if os.path.isfile('timesv2.txt') else 'w'
    with open('timesv2.txt', wmode) as f:
        f.write(f"{argsp.strategy}, gradient: {use_gradient} --> {result['time']}\n")
    exit()
