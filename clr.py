# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:33:29 2018

@author: NGN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def clr(base_lr, max_lr, step_size, clr_iterations, name=None):
    if clr_iterations is None:
        raise ValueError('global_step is required for clr!')
    with ops.name_scope(name, 'clr', [base_lr, max_lr, step_size, clr_iterations]) as name:
        base_lr = ops.convert_to_tensor(base_lr, name="learning_rate")
        dtype = base_lr.dtype
        max_lr = math_ops.cast(max_lr, dtype)
        step_size = math_ops.cast(step_size, dtype)
        clr_iterations = math_ops.cast(clr_iterations, dtype)
        cycle = math_ops.floor(1+clr_iterations/(2*step_size))
        x = math_ops.abs(clr_iterations/step_size - 2*cycle + 1)
        return math_ops.add(base_lr, math_ops.mul(math_ops.subtract(max_lr,base_lr),math_ops.maximum(0., math_ops.subtract(1.,x))))