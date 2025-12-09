import torch
import torch.nn as nn
import torch.nn.functional as F
from common.network import LayerType


def batch(A, n):
    return A.view(n, -1, *A.size()[1:])


def unbatch(A):
    return A.view(-1, *A.size()[2:])


class DualLinear_Ind():
    def __init__(self, layer):
        self.weight = layer.weight
        self.bias = layer.bias

    def T(self, inp_vs):
        inp_v = inp_vs[-1]
        if inp_v is None:
            return None
        new_inp_v = F.linear(inp_v, self.weight.t())

        return new_inp_v


class DualConv2D_Ind():
    def __init__(self, layer, in_shape, out_shape):
        self.in_shape = in_shape  # forward pass
        self.out_shape = out_shape  # forward pass
        self.weight = layer.weight
        self.stride = layer.stride
        self.padding = layer.padding
        self.group = 1
        self.dilation = layer.dilation
        # self.bias = layer.bias

        C_out, H_out, W_out = self.out_shape
        b = layer.bias.view(C_out, 1, 1).expand(C_out, H_out, W_out)
        self.bias = b.flatten()

    def conv_transpose2d(self, v):
        i = 0
        out = []
        batch_size = 10000

        kH, kW = self.weight.shape[-2:]
        C_out, out_H, out_W = self.out_shape
        C_in,  in_H,  in_W = self.in_shape

        # Compute output_padding so that transposed conv maps (out_H, out_W) -> (in_H, in_W)
        base_H = (out_H - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (kH - 1) + 1
        base_W = (out_W - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (kW - 1) + 1
        op_h = in_H - base_H
        op_w = in_W - base_W

        if not (0 <= op_h < self.stride[0] and 0 <= op_w < self.stride[1]):
            raise ValueError(f"Computed output_padding {(op_h, op_w)} invalid for stride {self.stride}. "
                             f"Check in/out shapes or conv params.")
        out_padding = (int(op_h), int(op_w))

        while i < v.size(0):
            out.append(F.conv_transpose2d(v[i:min(i+batch_size, v.size(0))], self.weight, None,
                                          self.stride, self.padding, out_padding, self.group, self.dilation))
            i += batch_size
        return torch.cat(out, 0)

    def conv_reshape(self, v):
        if v.dim() != 1:
            raise ValueError("Input tensor must be 1D for reshaping")
        C, H, W = self.out_shape
        if v.numel() != C * H * W:
            raise ValueError(f"Expected {C*H*W} elements, got {v.numel()}")
        # print(f"out_shape: {self.out_shape}, v: {v.size()}")
        # print(f"v.reshape(1, C, H, W): {v.reshape(1, C, H, W).size()}")
        return v.reshape(1, C, H, W)

    def T(self, inp_vs):
        inp_v = inp_vs[-1]
        if inp_v is None:
            return None
        inp_v = self.conv_reshape(inp_v)
        new_inp_v = self.conv_transpose2d(inp_v)
        new_inp_v = new_inp_v.flatten()

        return new_inp_v


class DualRelu_Ind():
    def __init__(self, inp_lb, inp_ub):
        self.inp_lb = inp_lb
        self.inp_ub = inp_ub

    def get_lambda(self, inp1_v, precise=False):
        inp_lb = self.inp_lb
        inp_ub = self.inp_ub

        inp_positive = (inp_lb >= 0) & (inp_ub > 0)
        inp_negative = (inp_ub <= 0)
        inp_unstable = ~(inp_positive) & ~(inp_negative)

        # $ preactivation state is negative
        lam = torch.zeros(inp_lb.size(), device=inp_lb.device)

        # $ preactivation state is positive
        lam = torch.where(inp_positive, torch.ones(inp_lb.size(), device=inp_lb.device), lam)

        # $ preactivation state is unstable
        temp_c = inp_ub / (inp_ub - inp_lb + 1e-15)
        slope_zero = -inp_lb >= inp_ub  # lower bound is y=0
        slope_one = inp_ub > -inp_lb    # upper bound is y=x
        if precise:
            lam = torch.where(inp_unstable & slope_zero & (inp1_v >= 0), temp_c, lam)
            lam = torch.where(inp_unstable & slope_one & (inp1_v >= 0), temp_c, lam)
            lam = torch.where(inp_unstable & slope_one & (inp1_v < 0), torch.ones_like(inp_lb), lam)
        else:
            lam = torch.where(inp_unstable, temp_c, lam)

        return lam

    def T(self, inp_vs):
        inp_v = inp_vs[-1]
        if inp_v is None:
            return None

        lam = self.get_lambda(inp_v)

        new_inp_v = lam * inp_v
        return new_inp_v
