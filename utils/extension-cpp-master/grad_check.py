from __future__ import division
from __future__ import print_function

import argparse
import torch

from torch.autograd import Variable, gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.lltm_baseline import LLTMFunction
elif options.example == 'cpp':
    from cpp.lltm import LLTMFunction
else:
    from cuda.lltm import LLTMFunction
    options.cuda = True

X = torch.randn(options.batch_size, options.features)
h = torch.randn(options.batch_size, options.state_size)
C = torch.randn(options.batch_size, options.state_size)
W = torch.randn(3 * options.state_size, options.features + options.state_size)
b = torch.randn(1, 3 * options.state_size)

variables = [X, W, b, h, C]

for i, var in enumerate(variables):
    if options.cuda:
        var = var.cuda()
    variables[i] = Variable(var.double(), requires_grad=True)

if gradcheck(LLTMFunction.apply, variables):
    print('Ok')
