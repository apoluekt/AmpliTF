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

#import tensorflow as tf

import sys, os
sys.path.append("../")
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import amplitf.interface as atfi

#atfi.backend_numpy()
#atfi.backend_jax()
#atfi.backend_tf()
atfi.backend_auto()

import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.toymc as atft
import amplitf.likelihood as atfl
import amplitf.optimisation as atfo
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace

#tf.config.experimental_run_functions_eagerly(False)

if __name__ == "__main__" : 

  # Four body angular phase space is described by 3 angles. 
  phsp = RectangularPhaseSpace(((-1., 1.), (-1., 1.), (-atfi.pi(), atfi.pi())))

  # Fit parameters of the model 
  FL  = atfi.FitParameter("FL" ,  0.770,  0.000, 1.000, 0.01)
  AT2 = atfi.FitParameter("AT2",  0.200, -1.000, 1.000, 0.01)
  S5  = atfi.FitParameter("S5" , -0.100, -1.000, 1.000, 0.01)

  pars = [ FL, AT2, S5 ]

  ### Start of model description

  #@atfi.function
  def model(x, params) : 
    FL  = params["FL"]
    AT2 = params["AT2"]
    S5  = params["S5"]

    # Get phase space variables
    cosThetaK = phsp.coordinate(x, 0)
    cosThetaL = phsp.coordinate(x, 1)
    phi = phsp.coordinate(x, 2)

    # Derived quantities
    sinThetaK = atfi.sqrt(1.0 - cosThetaK * cosThetaK)
    sinThetaL = atfi.sqrt(1.0 - cosThetaL * cosThetaL)

    sinTheta2K =  (1.0 - cosThetaK * cosThetaK)
    sinTheta2L =  (1.0 - cosThetaL * cosThetaL)

    sin2ThetaK = (2.0 * sinThetaK * cosThetaK)
    cos2ThetaL = (2.0 * cosThetaL * cosThetaL - 1.0)

    # Decay density
    pdf  = (3.0/4.0) * (1.0 - FL ) * sinTheta2K
    pdf +=  FL * cosThetaK * cosThetaK
    pdf +=  (1.0/4.0) * (1.0 - FL ) * sin2ThetaK *  cos2ThetaL
    pdf +=  (-1.0) * FL * cosThetaK * cosThetaK *  cos2ThetaL
    pdf +=  (1.0/2.0) * (1.0 - FL ) * AT2 * sinTheta2K * sinTheta2L * atfi.cos(2.0 * phi)
    pdf +=  S5 * sin2ThetaK * sinThetaL * atfi.cos(phi)

    return atfi.abs(pdf)
  ### End of model description

  initpars = { p.name : p.init_value for p in pars }

  @atfi.function
  def gen_model(x) : 
    return model(x, initpars)

  atfi.set_seed(1)

  # Estimate the maximum of PDF for toy MC generation using accept-reject method
  maximum = atft.maximum_estimator(gen_model, phsp, 100000) * 1.5
  print("Maximum = ", maximum)

  # Create toy MC data sample (with the model parameters set to their initial values)
  data_sample = atft.run_toymc(gen_model, phsp, 1000000, maximum, chunk = 1000000)

  print(data_sample)

  norm_sample = phsp.uniform_sample(1000000)

  # TF graph for unbinned negalite log likelihood (the quantity to be minimised)
  @atfi.function
  def nll(data, norm, pars) : 
    return atfl.unbinned_nll(model(data, pars), atfl.integral(model(norm, pars)))

  # Run MINUIT minimisation of the neg. log likelihood
  result = atfo.run_minuit(nll, pars, args = (data_sample, norm_sample), use_gradient = True)
  print(result)

  print(f"{result['time']/result['func_calls']} sec per function call")
