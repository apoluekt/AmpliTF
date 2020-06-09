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
import numpy as np
import amplitf.interface as atfi

from iminuit import Minuit

from timeit import default_timer as timer

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
        kwargs = { p.name : p() for p in float_pars }
        func.n += 1
        nll_val = nll(*args, kwargs)
        if func.n % 10 == 0 : print(func.n, nll_val, par)
        return nll_val

    gradient_supported = atfi.create_gradient(nll, args, float_pars)
    if not gradient_supported : 
        print("Backend does not support automatic gradient, reverting to numerical. ")

    func.n = 0
    atfi.gradient.n = 0

    start = [ p.init_value for p in float_pars ]
    error = [ p.step_size for p in float_pars ]
    limit = [ (p.lower_limit, p.upper_limit) for p in float_pars ]
    name = [ p.name for p in float_pars ]

    if use_gradient and gradient_supported : 
        minuit = Minuit.from_array_func(func, start, error = error, limit = limit, name = name, grad = atfi.gradient, errordef = 0.5)
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
        results["params"][p.name] = (p.fitted_value, p.error)

    # return fit results
    results["loglh"] = f_min.fval
    results["iterations"] = f_min.ncalls
    results["func_calls"] = func.n
    results["grad_calls"] = atfi.gradient.n
    results["time"] = end-start
    return results

def calculate_fit_fractions(pdf, norm_sample) :
    """
      Calculate fit fractions for PDF components
        norm_sample : normalisation sample. 
    """
    args, varargs, keywords, defaults = inspect.getargspec(pdf)
    num_switches = 0
    if defaults : 
      default_dict = dict(zip(args[-len(defaults):], defaults))
      if "switches" in default_dict : num_switches = len(default_dict["switches"])

    @atfi.function
    def pdf_components(d) : 
        result = []
        for i in range(num_switches) : 
            switches = num_switches*[ 0 ]
            switches[i] = 1
            result += [ pdf(d, switches = switches) ]
        return result

    total_int = atfi.reduce_sum(pdf(norm_sample))
    return [ atfi.reduce_sum(i)/total_int for i in pdf_components(norm_sample) ]

def write_fit_results(pars, results, filename) :
    """
      Write the dictionary of fit results to text file
        results : fit results as returned by MinuitFit
        filename : file name
    """
    f = open(filename, "w")
    for p in pars :
        if not p.name in results["params"] : continue
        s = "%s " % p.par_name
        for i in results["params"][p.par_name]:
            s += "%f " % i
        f.write(s + "\n")
    s = "loglh %f %f" % (results["loglh"], results["initlh"])
    f.write(s + "\n")
    f.close()
