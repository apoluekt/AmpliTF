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
import amplitf.interface as atfi


@atfi.function
def integral(pdf):
    """
      Return the graph for the integral of the PDF
        pdf : PDF
    """
    return tf.reduce_mean(pdf)


@atfi.function
def weighted_integral(pdf, weight_func):
    """
      Return the graph for the integral of the PDF
        pdf : PDF
        weight_func : weight function
    """
    return tf.reduce_mean(pdf*weight_func)


@atfi.function
def unbinned_nll(pdf, integral):
    """
      Return unbinned negative log likelihood graph for a PDF
        pdf      : PDF 
        integral : precalculated integral
    """
    return -tf.reduce_sum(atfi.log(pdf / integral))


@atfi.function
def unbinned_weighted_nll(pdf, integral, weight_func):
    """
      Return unbinned weighted negative log likelihood graph for a PDF
        pdf         : PDF
        integral    : precalculated integral
        weight_func : weight function
    """
    return -tf.reduce_sum(atfi.log(pdf / integral) * weight_func)
