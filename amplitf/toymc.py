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

def accept_reject_sample(density, sample):
    """
      Return toy MC sample graph using accept-reject method
        density : function to calculate density
        sample  : input uniformly distributed sample
    """
    x = sample[..., 0:-1]
    if density is not None :
        r = sample[..., -1]
        return x[density(x) > r]
    else:
        return x

@atfi.generator_function
def maximum_estimator(density, phsp, size):
    """
      Return the graph for the estimator of the maximum of density function
        density : density function
        phsp : phase space object (should have uniform_sample method implemented)
        size : size of the random sample for maximum estimation
    """
    sample = phsp.uniform_sample(size)
    return atfi.reduce_max(density(sample))

def run_toymc(pdf, phsp, size, maximum, chunk=200000, seed=None, components = True):
    """
      Create toy MC sample. To save memory, the sample is generated in "chunks" of a fixed size 
             pdf : Function returning PDF graph for a given sample as an agrument
            phsp : phase space
            size : size of the target data sample (if >0) or number of chunks (if <0)
         maximum : maximum PDF value for accept-reject method
           chunk : chunk size
            seed : initial random seed. Not initalised if None
    """
    import inspect
    length, nchunk, curr_maximum = 0, 0, maximum
    dim = phsp.dimensionality()
    data = None

    if seed is not None : 
        atfi.set_seed(seed)

    def condition(length, size, nchunk):
        return length < size or nchunk < -size

    @atfi.generator_function
    def pdf_vals(chunk, curr_maximum) : 
        d = accept_reject_sample(pdf, phsp.filter(phsp.unfiltered_sample(chunk, curr_maximum)))
        return d, pdf(d)

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
            result += [ pdf(d, switches = tuple(switches) ) ]
        return result

    while condition(length, size, nchunk):
        d,v = pdf_vals(chunk, curr_maximum)
        over_maximum = v[v > curr_maximum]
        if len(over_maximum) > 0:
            new_maximum = atfi.reduce_max(over_maximum)*1.5
            print(f'  Updating maximum: {curr_maximum} -> {new_maximum}. Starting over.')
            length, nchunk, curr_maximum = 0, 0, new_maximum
            data = None
            continue
        if components and num_switches > 0 : 
            vs = pdf_components(d)
            wd = atfi.stack([i/v for i in vs], axis=1)
            d = atfi.concat([d, wd], axis=1)
        if data is not None : 
            data = atfi.concat([data, d], axis=0)
        else : 
            data = d
        length += len(d)
        nchunk += 1
        print(f'  Chunk {nchunk}, size={len(d)}, total length={length}')
    return data[:size] if size > 0 else data
