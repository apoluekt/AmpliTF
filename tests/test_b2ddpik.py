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

import sys, os
sys.path.append("../")
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.toymc as atft
import amplitf.rootio as atfr
from amplitf.phasespace.b2ddkpi_phasespace import B2DDKpiPhaseSpace
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace
import amplitf.dalitz_decomposition as atfdd

import argparse

if __name__ == "__main__" : 

  mb   = 5.27932
  mdst = 2.00685
  mdz  = 1.86483
  mk   = 0.493677
  mpi  = 0.13957061
  mx   = 3.87169
  mxmax = 2.010+1.869
  gamma_star_z = 1.*65.5e-6
  mu = mdst*mdz/(mdst + mdz)
  I = atfi.complex(atfi.const(0.), atfi.const(1.))

  parser = argparse.ArgumentParser(description = "B2X(DD*)K toy MC")
  parser.add_argument('--seed', type=int, default = 0, 
                      help="Initial random seed")
  parser.add_argument('--size', type=int, default = 10000, 
                      help="Number of events to generate")
  parser.add_argument('--maximum', type=float, default = 1., 
                      help="Maximum of PDF for rejection sampling")
  parser.add_argument('--output', type=str, default = "output.root", 
                      help="Output ROOT file")
  parser.add_argument('--reg', type=float, default = 0., 
                      help="Re(gamma), typically -50<Re(gamma)<50")
  parser.add_argument('--img', type=float, default = 0., 
                      help="Im(gamma), typically 0<Im(gamma)<30")
  parser.add_argument('--chunk', type=int, default = 1000000, 
                      help="Chunk size")
  parser.add_argument('--model', type=str, default = "braaten",  
                      help="X lineshape model name")
  parser.add_argument('--gamma', default = False, action = "store_const", const = True, 
                    help='Generate D*->Dgamma mode (default is D*->Dpi0)')
  parser.add_argument('--randomise', default = False, action = "store_const", const = True, 
                    help='Randomise model parameters')

  args = parser.parse_args()

  if len(sys.argv)<2 : 
    parser.print_help()
    raise SystemExit

  randomise_params = args.randomise
  modelname = args.model
  events = args.size
  gamma = args.gamma
  chunk = args.chunk
  reg = args.reg
  img = args.img

  # Four body angular phase space is described by 3 angles. 
  phsp_pi = B2DDKpiPhaseSpace(mdz, mk, mpi, mb, mddpirange = (2.*mdz + mpi, mxmax) )
  phsp_gm = B2DDKpiPhaseSpace(mdz, mk, 0., mb,  mddpirange = (2.*mdz, mxmax) )

  @atfi.function
  def helicity(x, nu, lmbd, phsp) :
    s = phsp.s(x)
    sigma1 = phsp.sigma1(x)
    sigma3 = phsp.sigma3(x)
    sigma2 = phsp.sigma2(x)
    theta1 = atfi.acos( phsp.costheta1(x) )
    phi23 = phsp.phi23(x)
    sqrts = atfi.sqrt(s)
    theta12 = atfi.acos( atfdd.cos_theta_12(sqrts, mdz, phsp.mpi, mdz, sigma1, sigma2, sigma3) )
    theta23 = atfi.acos( atfdd.cos_theta_23(sqrts, mdz, phsp.mpi, mdz, sigma1, sigma2, sigma3) )
    theta_hat_3_canonical_1 = atfi.acos( atfdd.cos_theta_hat_3_canonical_1(sqrts, mdz, phsp.mpi, mdz, sigma1, sigma2, sigma3) )
    cap_d = atfk.wigner_capital_d(atfi.zeros(phi23), theta1, phi23, 2, 0, nu)
    hel1 = cap_d*atfi.cast_complex(atfk.wigner_small_d(theta23, 2, nu, lmbd))
    hel2 = atfi.cast_complex(atfi.const(0.))
    for tau in [ -2, 0, 2] :
      hel2 += atfi.cast_complex(atfk.wigner_small_d(theta_hat_3_canonical_1, 2, nu, tau)*\
                                atfk.wigner_small_d(theta12, 2, tau, lmbd))*cap_d*\
              atfi.clebsch(2, 0, 0, 0, 2, 0)*atfi.clebsch(0, 0, 2, nu, 2, nu)
    return hel1, hel2

  @atfi.function
  def dstar_bw(sigma1, sigma3) :
    return (atfd.relativistic_breit_wigner(sigma1, atfi.const(mdst), atfi.const(gamma_star_z)), 
            atfd.relativistic_breit_wigner(sigma3, atfi.const(mdst), atfi.const(gamma_star_z)) )

  @atfi.function
  def braaten_lineshape(s, reg, img) :
    e = atfi.cast_complex(atfi.sqrt(s) - mdst - mdz)
    gamma = atfi.complex(reg*1e-3, img*1e-3)
    return 1./(-gamma + atfi.sqrt(-2.*mu*( e + I*gamma_star_z/2. ) ) )

  def build_model(lineshape, args, randomise_params = False, gamma = False) :

    if gamma : 
      phsp = phsp_gm
      lmbds = [-2, 2]
    else : 
      phsp = phsp_pi 
      lmbds = [0]

    @tf.function(autograph=False)
    def model(x) : 
      s = phsp.s(x)
      sigma1 = phsp.sigma1(x)
      sigma3 = phsp.sigma3(x)
      if randomise_params : par = { a[0] : x[:,5+i] for i,a in enumerate(args) }
      else : par = { a[0] : atfi.const(a[1]) for a in args }
      ls = lineshape(s, **par)
      bw1, bw2 = dstar_bw(sigma1, sigma3)
      dens = atfi.const(0.)
      for lmbd in lmbds : 
        hel = atfi.cast_complex(atfi.const(0))
        for nu in [-2, 0, 2] : 
          hel1, hel2 = helicity(x, nu, lmbd, phsp)
          hel += hel1*bw1 + hel2*bw2
        dens += atfi.density( ls*hel )
      return phsp.density(x)*dens

    return model, phsp

  ### End of model description

  @atfi.function
  def observables(x, phsp) : 
    p4d1, p4d2, p4k, p4pi = phsp.final_state_momenta(x)
    mdd  = atfk.mass( p4d1 + p4d2 )
    mddk = atfk.mass( p4d1 + p4d2 + p4k )
    md1k = atfk.mass( p4d1 + p4k )
    md2k = atfk.mass( p4d2 + p4k )
    hel = atfk.cos_helicity_angle_dalitz( mdd**2, md1k**2, mddk, mdz, mdz, mk )
    pd = 1000.*mdd**2/(2.*mdz)*atfi.sqrt(1-4.*(mdz**2)/mdd**2)
    return tf.stack( [mdd, mddk, md1k, md2k, hel, pd], axis = 1 )

  atfi.set_seed(1)

  models = {
    "braaten" : {
      "lineshape" : braaten_lineshape, 
      "pars" : [ ("reg", reg), ("img",  img) ], 
      "phsp" : RectangularPhaseSpace( ((-50., 50.), (0., 30.)) )
    }
  }

  model, xphsp = build_model(models[modelname]["lineshape"], models[modelname]["pars"], 
                             gamma = gamma, randomise_params = randomise_params)
  if randomise_params : 
    phsp = CombinedPhaseSpace(xphsp, models[modelname]["phsp"])
  else : 
    phsp = xphsp

  print(phsp.dimensionality(), phsp.bounds() )

  atfi.set_seed(args.seed)

  # Estimate the maximum of PDF for toy MC generation using accept-reject method
  maximum = atft.maximum_estimator(model, phsp, 100000) * 1.5
  print("Maximum = ", maximum)

  # Create toy MC data sample (with the model parameters set to their initial values)
  data_sample = atft.run_toymc(model, phsp, events, maximum, chunk = chunk)
  obs_array = observables(data_sample, xphsp)

  array = np.concatenate( [data_sample, obs_array], axis = 1 )
  #array = data_sample

  branches = ["s", "sigma1", "sigma3", "costheta1", "phi23"]
  if randomise_params : 
       branches += [ p[0] for p in models[modelname]["pars"] ]
  branches += [ "mdd", "mddk", "md2k", "md2k", "hel", "pd" ]

  atfr.write_tuple("test.root", array, branches )
