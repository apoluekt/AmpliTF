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
import amplitf.interface as atfi

from iminuit import Minuit

from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from timeit import default_timer as timer

class FitParameter(ResourceVariable) : 
    def __init__(self, name, init_value, lower_limit, upper_limit, step_size = 1e-6) : 
        global __all_variables__
        ResourceVariable.__init__(self, init_value, dtype = atfi.fptype(), trainable = True)
        self.init_value = init_value
        self.par_name = name
        self.step_size = step_size
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.prev_value = None
        self.fixed = False
        self.error = 0.
        self.positive_error = 0.
        self.negative_error = 0.
        self.fitted_value = init_value

    def update(self, value) : 
        if value != self.prev_value : 
            self.assign(value)
            self.prev_value = value

    def fix(self):
        self.fixed = True

    def float(self):
        self.fixed = False

    def setFixed(self, fixed):
        self.fixed = fixed

    def floating(self):
        """
          Return True if the parameter is floating and step size>0
        """
        return self.step_size > 0 and not self.fixed

def run_minuit(nll, pars, args, use_gradient = True) :
    """
      Run IMinuit to minimise NLL function

      nll  : python callable representing the negative log likelihood to be minimised
      pars : list of FitParameters
      args : arguments of the nll callable (typically, data and/or normalisation samples)
      use_gradient : if True, use analytic gradient

      returns the dictionary with the values and errors of the fit parameters
    """

    float_pars = [ p for p in pars if p.floating() ]

    def func(par) :
        for i,p in enumerate(float_pars) : p.update(par[i])
        func.n += 1
        nll_val = nll(*args).numpy()
        if func.n % 10 == 0 : print(func.n, nll_val, [ i.numpy() for i in float_pars ] )
        return nll_val

    def grad(par) : 
        for i,p in enumerate(float_pars) : p.update(par[i])
        grad.n += 1
        with tf.GradientTape() as gradient : 
            gradient.watch(float_pars)
            nll_val = nll(*args)
        g = gradient.gradient(nll_val, float_pars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        g_val = [ i.numpy() for i in g ]
        return g_val

    func.n = 0
    grad.n = 0

    start = [ p.init_value for p in float_pars ]
    error = [ p.step_size for p in float_pars ]
    limit = [ (p.lower_limit, p.upper_limit) for p in float_pars ]
    name = [ p.par_name for p in float_pars ]

    if use_gradient : 
      minuit = Minuit.from_array_func(func, start, error = error, limit = limit, name = name, grad = grad, errordef = 0.5)
    else : 
      minuit = Minuit.from_array_func(func, start, error = error, limit = limit, name = name, errordef = 0.5)

    start = timer()
    minuit.migrad()
    end = timer()

    par_states = minuit.get_param_states()
    f_min = minuit.get_fmin()

    results = { "params" : {} } # Get fit results and update parameters
    for n, p in enumerate(float_pars) :
        p.update(par_states[n].value)
        p.fitted_value = par_states[n].value
        p.error = par_states[n].error
        results["params"][p.par_name] = (p.fitted_value, p.error)

    # return fit results
    results["loglh"] = f_min.fval
    results["iterations"] = f_min.ncalls
    results["func_calls"] = func.n
    results["grad_calls"] = grad.n
    results["time"] = end-start
    return results
