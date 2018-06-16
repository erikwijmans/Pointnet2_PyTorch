import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

torch.manual_seed(42)


def d_sigmoid(z):
    s = F.sigmoid(z)
    return (1 - s) * s


def d_tanh(z):
    t = F.tanh(z)
    return 1 - (t * t)


def d_elu(z, alpha=1.0):
    e = z.exp()
    mask = (alpha * (e - 1)) < 0
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e)


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        X = torch.cat([old_h, input], dim=1)

        gate_weights = F.linear(X, weights, bias)
        gates = gate_weights.chunk(3, dim=1)

        input_gate = F.sigmoid(gates[0])
        output_gate = F.sigmoid(gates[1])
        candidate_cell = F.elu(gates[2])

        new_cell = old_cell + candidate_cell * input_gate
        new_h = F.tanh(new_cell) * output_gate

        ctx.save_for_backward(X, weights, input_gate, output_gate, old_cell,
                              new_cell, candidate_cell, gate_weights)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        X, weights, input_gate, output_gate, old_cell = ctx.saved_variables[:5]
        new_cell, candidate_cell, gate_weights = ctx.saved_variables[5:]

        d_input = d_weights = d_bias = d_old_h = d_old_cell = None

        d_output_gate = F.tanh(new_cell) * grad_h
        d_tanh_new_cell = output_gate * grad_h
        d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell

        d_old_cell = d_new_cell
        d_candidate_cell = input_gate * d_new_cell
        d_input_gate = candidate_cell * d_new_cell

        gates = gate_weights.chunk(3, dim=1)
        d_input_gate *= d_sigmoid(gates[0])
        d_output_gate *= d_sigmoid(gates[1])
        d_candidate_cell *= d_elu(gates[2])

        d_gates = torch.cat(
            [d_input_gate, d_output_gate, d_candidate_cell], dim=1)

        if ctx.needs_input_grad[1]:
            d_weights = d_gates.t().mm(X)
        if ctx.needs_input_grad[2]:
            d_bias = d_gates.sum(dim=0, keepdim=True)
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            d_X = d_gates.mm(weights)
            state_size = grad_h.shape[1]
            d_old_h, d_input = d_X[:, :state_size], d_X[:, state_size:]

        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
