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

"""Common neural network activations and other functions.
"""

from __future__ import absolute_import
from __future__ import division

import numpy as onp

from jax import lax
from jax import random
from jax.scipy.special import logsumexp, expit
import jax.numpy as np

def relu(x): return np.maximum(x, 0.)
def softplus(x): return np.logaddexp(x, 0.)
def sigmoid(x): return expit(x)
def elu(x): return np.where(x > 0, x, np.exp(x) - 1)
def leaky_relu(x): return np.where(x >= 0, x, 0.01 * x)

def logsoftmax(x, axis=-1):
  """Apply log softmax to an array of logits, log-normalizing along an axis."""
  return x - logsumexp(x, axis, keepdims=True)

def softmax(x, axis=-1):
  """Apply softmax to an array of logits, exponentiating and normalizing along an axis."""
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)

def fastvar(x, axis, keepdims):
  """A fast but less numerically-stable variance calculation than np.var."""
  return np.mean(x**2, axis, keepdims=keepdims) - np.mean(x, axis, keepdims=keepdims)**2
