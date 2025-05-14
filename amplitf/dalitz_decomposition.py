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

"""
  Formulas for Dalitz plot decomposition angles from M. Mikhasenko et al. https://arxiv.org/pdf/1910.04566.pdf
"""

from amplitf.interface import sqrt


def kallen_function(x, y, z):
    """
    Kallen's "lambda" function
    """
    return x ** 2 + y ** 2 + z ** 2 - 2 * x * y - 2 * x * z - 2 * y * z


def cos_theta_12(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return (
        2 * sigma3 * (sigma2 - m3 ** 2 - m1 ** 2)
        - (sigma3 + m1 ** 2 - m2 ** 2) * (M ** 2 - sigma3 - m3 ** 2)
    ) / sqrt(
        kallen_function(M ** 2, m3 ** 2, sigma3)
        * kallen_function(sigma3, m1 ** 2, m2 ** 2)
    )


def cos_theta_23(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_theta_12(M, m2, m3, m1, sigma2, sigma3, sigma1)


def cos_theta_31(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_theta_12(M, m3, m1, m2, sigma3, sigma1, sigma2)


def cos_theta_hat_3_canonical_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return (
        (M ** 2 + m3 ** 2 - sigma3) * (M ** 2 + m1 ** 2 - sigma1)
        - 2 * M ** 2 * (sigma2 - m3 ** 2 - m1 ** 2)
    ) / sqrt(
        kallen_function(M ** 2, m1 ** 2, sigma1)
        * kallen_function(M ** 2, sigma3, m3 ** 2)
    )


def cos_theta_hat_1_canonical_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_theta_hat_3_canonical_1(M, m2, m3, m1, sigma2, sigma3, sigma1)


def cos_theta_hat_2_canonical_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_theta_hat_3_canonical_1(M, m3, m1, m2, sigma3, sigma1, sigma2)


def cos_zeta_1_aligned_3_in_frame_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return (
        2 * m1 ** 2 * (sigma2 - M ** 2 - m2 ** 2)
        + (M ** 2 + m1 ** 2 - sigma1) * (sigma3 - m1 ** 2 - m2 ** 2)
    ) / sqrt(
        kallen_function(M ** 2, m1 ** 2, sigma1)
        * kallen_function(sigma3, m1 ** 2, m2 ** 2)
    )


def cos_zeta_1_aligned_1_in_frame_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_zeta_1_aligned_3_in_frame_1(M, m1, m3, m2, sigma1, sigma3, sigma2)


def cos_zeta_2_aligned_1_in_frame_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_zeta_1_aligned_3_in_frame_1(M, m2, m3, m1, sigma2, sigma3, sigma1)


def cos_zeta_2_aligned_2_in_frame_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_zeta_1_aligned_3_in_frame_1(M, m2, m1, m3, sigma2, sigma1, sigma3)


def cos_zeta_3_aligned_3_in_frame_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_zeta_1_aligned_3_in_frame_1(M, m3, m1, m2, sigma3, sigma1, sigma2)


def cos_zeta_3_aligned_2_in_frame_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
    return cos_zeta_1_aligned_3_in_frame_1(M, m3, m2, m1, sigma3, sigma2, sigma1)
