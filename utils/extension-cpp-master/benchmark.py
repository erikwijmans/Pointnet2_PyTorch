from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-s', '--state-size', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.lltm import LLTM
elif options.example == 'cpp':
    from cpp.lltm import LLTM
else:
    from cuda.lltm import LLTM
    options.cuda = True

X = torch.randn(options.batch_size, options.features)
h = torch.randn(options.batch_size, options.state_size)
C = torch.randn(options.batch_size, options.state_size)
rnn = LLTM(options.features, options.state_size)

if options.cuda:
    X = X.cuda()
    h = h.cuda()
    C = C.cuda()
    rnn.cuda()

# Force CUDA initialization
new_h, new_C = rnn(X, (h, C))
(new_h.sum() + new_C.sum()).backward()

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0
for _ in range(options.runs):
    rnn.zero_grad()

    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))
