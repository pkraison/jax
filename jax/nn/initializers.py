# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""

from __future__ import absolute_import
from __future__ import division

import numpy as onp

from jax import lax
from jax import random
import jax.numpy as np

def zeros(key, shape, dtype=np.float32): return np.zeros(shape, dtype)
def ones(key, shape, dtype=np.float32): return np.ones(shape, dtype)

def uniform(scale=1e-2):
  def init(key, shape, dtype=np.float32):
    return random.uniform(key, shape, dtype) * scale
  return init

def normal(stddev=1e-2):
  def init(key, shape, dtype=np.float32):
    return random.normal(key, shape, dtype) * stddev
  return init

def _compute_fans(shape):
  assert len(shape) >= 2
  receptive_field_size = onp.prod(shape[:-2])
  fan_in = shape[-2] * receptive_field_size
  fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out

def variance_scaling(scale, mode, distribution):
  def init(key, shape, dtype=np.float32):
    fan_in, fan_out = _compute_fans(shape)
    if mode == "fan_in": scale /= fan_in
    elif mode == "fan_out": scale /= fan_out
    elif mode == "fan_avg": scale /= (fan_in + fan_out) / 2
    else: raise ValueError("invalid mode for variance scaling initializer")
    if distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = onp.sqrt(scale) / .87962566103423978
      return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return random.normal(key, shape, dtype) * onp.sqrt(scale)
    elif distribution == "uniform":
      return random.uniform(key, shape, dtype, -1) * onp.sqrt(3 * scale)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
    return init

glorot_uniform = variance_scaling(1.0, "fan_avg", "uniform")
glorot_normal = variance_scaling(1.0, "fan_avg", "truncated_normal")
lecun_uniform = variance_scaling(1.0, "fan_in", "uniform")
lecun_normal = variance_scaling(1.0, "fan_in", "truncated_normal")
kaiming_uniform = he_uniform = variance_scaling(2.0, "fan_in", "uniform")
kaiming_normal = he_normal = variance_scaling(2.0, "fan_in", "truncated_normal")