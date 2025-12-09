import torch
import torch.nn.functional as F
from common.network import LayerType


def select_layer(layer, pre_conv=None, post_conv=None, pre_lb=None, pre_ub=None):
    if layer.type is LayerType.Linear:
        return DualLinear(layer)
    elif layer.type is LayerType.Conv2D:
        if pre_conv is None or post_conv is None:
            raise ValueError("pre_conv and post_conv must be provided for DualConv2D")
        return DualConv2D(layer, pre_shape=pre_conv, post_shape=post_conv)
    elif layer.type is LayerType.ReLU:
        if pre_lb is None or pre_ub is None:
            raise ValueError("pre_lb and pre_ub must be provided for DualRelu")
        return DualRelu(pre_lb, pre_ub)
    else:
        raise ValueError(f"Unsupported layer type: {layer.type}")


def batch(A, n):
    return A.view(n, -1, *A.size()[1:])


def unbatch(A):
    return A.view(-1, *A.size()[2:])


class DualLinear():
    def __init__(self, layer):
        self.weight = layer.weight
        self.bias = layer.bias

    def T(self, inp1_vs, inp2_vs, delta_vs):
        inp1_v = inp1_vs[-1]
        inp2_v = inp2_vs[-1]
        delta_v = delta_vs[-1]
        if inp1_v is None or inp2_v is None or delta_v is None:
            return None
        new_inp1_v = F.linear(inp1_v, self.weight.t())
        new_inp2_v = F.linear(inp2_v, self.weight.t())
        new_delta_v = F.linear(delta_v, self.weight.t())

        return new_inp1_v, new_inp2_v, new_delta_v


class DualConv2D():
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

    def T(self, inp1_vs, inp2_vs, delta_vs):
        inp1_v = inp1_vs[-1]
        inp2_v = inp2_vs[-1]
        delta_v = delta_vs[-1]
        if inp1_v is None or inp2_v is None or delta_v is None:
            return None
        inp1_v = self.conv_reshape(inp1_v)
        inp2_v = self.conv_reshape(inp2_v)
        delta_v = self.conv_reshape(delta_v)
        new_inp1_v = self.conv_transpose2d(inp1_v)
        new_inp2_v = self.conv_transpose2d(inp2_v)
        new_delta_v = self.conv_transpose2d(delta_v)
        new_inp1_v = new_inp1_v.flatten()
        new_inp2_v = new_inp2_v.flatten()
        new_delta_v = new_delta_v.flatten()

        return new_inp1_v, new_inp2_v, new_delta_v


class DualRelu():
    def __init__(self, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub):
        self.inp1_lb = inp1_lb
        self.inp1_ub = inp1_ub
        self.inp2_lb = inp2_lb
        self.inp2_ub = inp2_ub
        self.d_lb = d_lb
        self.d_ub = d_ub

    def get_lambda(self, inp1_v, inp2_v, delta_v, precise=False):
        inp1_lb = self.inp1_lb
        inp1_ub = self.inp1_ub
        inp2_lb = self.inp2_lb
        inp2_ub = self.inp2_ub
        d_lb = self.d_lb
        d_ub = self.d_ub

        inp1_positive = (inp1_lb >= 0) & (inp1_ub > 0)
        inp1_negative = (inp1_ub <= 0)
        inp1_unstable = ~(inp1_positive) & ~(inp1_negative)
        inp2_positive = (inp2_lb >= 0) & (inp2_ub > 0)
        inp2_negative = (inp2_ub <= 0)
        inp2_unstable = ~(inp2_positive) & ~(inp2_negative)
        delta_positive = (d_lb >= 0) & (d_ub > 0)
        delta_negative = (d_ub <= 0)
        delta_unstable = ~(delta_positive) & ~(delta_negative)

        # $ (case 1) inp1 is negative, inp2 is negative
        l_inp1 = torch.zeros(inp1_lb.size(), device=inp1_lb.device)
        l_inp1_d = torch.zeros(inp1_lb.size(), device=inp1_lb.device)
        l_inp2 = torch.zeros(inp2_lb.size(), device=inp2_lb.device)
        l_inp2_d = torch.zeros(inp2_lb.size(), device=inp2_lb.device)
        l_d = torch.zeros(d_lb.size(), device=d_lb.device)
        # $ (case 2) inp1 is positive, inp2 is negative
        case_2 = inp1_positive & inp2_negative
        l_inp1 = torch.where(case_2, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1)
        l_inp1_d = torch.where(case_2, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1_d)
        # $ (case 3) inp1 is negative, inp2 is positive
        case_3 = inp1_negative & inp2_positive
        l_inp2 = torch.where(case_3, torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2)
        l_inp2_d = torch.where(case_3, -torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2_d)
        # $ (case 4) inp1 is positive, inp2 is positive
        case_4 = inp1_positive & inp2_positive
        l_inp1 = torch.where(case_4, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1)
        l_inp1_d = torch.where(case_4, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1_d)
        l_inp2 = torch.where(case_4, torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2)
        l_inp2_d = torch.where(case_4, -torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2_d)
        if inp1_unstable.any() or inp2_unstable.any():
            temp_inp1 = inp1_ub / (inp1_ub - inp1_lb + 1e-15)
            temp_inp2 = inp2_ub / (inp2_ub - inp2_lb + 1e-15)
            temp_d = d_ub / (d_ub - d_lb + 1e-15)
            inp1_v_pos = inp1_v >= 0
            inp1_v_neg = inp1_v < 0
            inp2_v_pos = inp2_v >= 0
            inp2_v_neg = inp2_v < 0
            inp1_zero = -inp1_lb >= inp1_ub  # lower bound is y=0
            inp1_one = inp1_ub > -inp1_lb  # upper bound is y=x
            inp2_zero = -inp2_lb >= inp2_ub  # lower bound is y=0
            inp2_one = inp2_ub > -inp2_lb  # upper bound is y=x
            # $ (case 5) inp1 is unstable, inp2 is negative
            case_5 = inp1_unstable & inp2_negative
            if precise:
                l_inp1 = torch.where(case_5 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                l_inp1 = torch.where(case_5 & inp1_one & inp1_v_neg, torch.ones_like(inp1_lb), l_inp1)
                l_inp1 = torch.where(case_5 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
            else:
                l_inp1 = torch.where(case_5, temp_inp1, l_inp1)
            l_inp1_d = torch.where(case_5, temp_inp1, l_inp1_d)
            # $ (case 6) inp1 is negative, inp2 is unstable
            case_6 = inp1_negative & inp2_unstable
            if precise:
                l_inp2 = torch.where(case_6 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
                l_inp2 = torch.where(case_6 & inp2_one & inp2_v_pos, torch.ones_like(inp2_lb), l_inp2)
                l_inp2 = torch.where(case_6 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
            else:
                l_inp2 = torch.where(case_6, temp_inp2, l_inp2)
            l_inp2_d = torch.where(case_6, -temp_inp2, l_inp2_d)
            # $ (case 7) inp1 is unstable, inp2 is positive
            case_7 = inp1_unstable & inp2_positive
            if precise:
                l_inp1 = torch.where(case_7 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                l_inp1 = torch.where(case_7 & inp1_one & inp1_v_neg, torch.ones_like(inp1_lb), l_inp1)
                l_inp1 = torch.where(case_7 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
            else:
                l_inp1 = torch.where(case_7, temp_inp1, l_inp1)
            l_inp1_d = torch.where(case_7, temp_inp1, l_inp1_d)
            l_inp2 = torch.where(case_7, torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2)
            l_inp2_d = torch.where(case_7, -torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2_d)
            # $ (case 8) inp1 is positive, inp2 is unstable
            case_8 = inp1_positive & inp2_unstable
            l_inp1 = torch.where(case_8, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1)
            l_inp1_d = torch.where(case_8, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1_d)
            if precise:
                l_inp2 = torch.where(case_8 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
                l_inp2 = torch.where(case_8 & inp2_one & inp2_v_pos, torch.ones_like(inp2_lb), l_inp2)
                l_inp2 = torch.where(case_8 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
            else:
                l_inp2 = torch.where(case_8, temp_inp2, l_inp2)
            l_inp2_d = torch.where(case_8, -temp_inp2, l_inp2_d)
            # $ (case 9) inp1 is unstable, inp2 is unstable, delta is positive
            case_9 = inp1_unstable & inp2_unstable & delta_positive
            if precise:
                l_inp1 = torch.where(case_9 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                l_inp1 = torch.where(case_9 & inp1_one & inp1_v_neg, torch.ones_like(inp1_lb), l_inp1)
                l_inp1 = torch.where(case_9 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                l_inp2 = torch.where(case_9 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
                l_inp2 = torch.where(case_9 & inp2_one & inp2_v_pos, torch.ones_like(inp2_lb), l_inp2)
                l_inp2 = torch.where(case_9 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
            else:
                l_inp1 = torch.where(case_9, temp_inp1, l_inp1)
                l_inp2 = torch.where(case_9, temp_inp2, l_inp2)
            case_9_pre_positive = case_9 & (delta_v > 0)
            l_d = torch.where(case_9_pre_positive, torch.ones(d_lb.size(), device=d_lb.device), l_d)
            # $ (case 10) inp1 is unstable, inp2 is unstable, delta is negative
            case_10 = inp1_unstable & inp2_unstable & delta_negative
            if precise:
                l_inp1 = torch.where(case_10 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                l_inp1 = torch.where(case_10 & inp1_one & inp1_v_neg, torch.ones_like(inp1_lb), l_inp1)
                l_inp1 = torch.where(case_10 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                l_inp2 = torch.where(case_10 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
                l_inp2 = torch.where(case_10 & inp2_one & inp2_v_pos, torch.ones_like(inp2_lb), l_inp2)
                l_inp2 = torch.where(case_10 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
            else:
                l_inp1 = torch.where(case_10, temp_inp1, l_inp1)
                l_inp2 = torch.where(case_10, temp_inp2, l_inp2)
            case_10_pre_negative = case_10 & (delta_v < 0)
            l_d = torch.where(case_10_pre_negative, torch.ones(d_lb.size(), device=d_lb.device), l_d)
            # $ (case 11) inp1 is unstable, inp2 is unstable, delta is unstable
            case_11 = inp1_unstable & inp2_unstable & delta_unstable
            if case_11.any():
                if precise:
                    l_inp1 = torch.where(case_11 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                    l_inp1 = torch.where(case_11 & inp1_one & inp1_v_neg, torch.ones_like(inp1_lb), l_inp1)
                    l_inp1 = torch.where(case_11 & inp1_zero & inp1_v_pos, temp_inp1, l_inp1)
                    l_inp2 = torch.where(case_11 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
                    l_inp2 = torch.where(case_11 & inp2_one & inp2_v_pos, torch.ones_like(inp2_lb), l_inp2)
                    l_inp2 = torch.where(case_11 & inp2_zero & inp2_v_neg, temp_inp2, l_inp2)
                else:
                    l_inp1 = torch.where(case_11, temp_inp1, l_inp1)
                    l_inp2 = torch.where(case_11, temp_inp2, l_inp2)
                case_11_pre_negative = case_11 & (delta_v < 0)
                case_11_pre_positive = case_11 & (delta_v > 0)
                temp_d_lb = -d_lb / (d_ub - d_lb + 1e-15)
                temp_d_ub = d_ub / (d_ub - d_lb + 1e-15)
                l_d = torch.where(case_11_pre_negative, temp_d_lb, l_d)
                l_d = torch.where(case_11_pre_positive, temp_d_ub, l_d)

        return l_inp1, l_inp1_d, l_inp2, l_inp2_d, l_d

    def T(self, inp1_vs, inp2_vs, delta_vs):
        inp1_v = inp1_vs[-1]
        inp2_v = inp2_vs[-1]
        delta_v = delta_vs[-1]
        if inp1_v is None:
            return None

        l_inp1, l_inp1_d, l_inp2, l_inp2_d, l_d = self.get_lambda(inp1_v, inp2_v, delta_v)

        new_inp1_v = l_inp1 * inp1_v + l_inp1_d * delta_v
        new_inp2_v = l_inp2 * inp2_v + l_inp2_d * delta_v
        new_delta_v = l_d * delta_v
        return new_inp1_v, new_inp2_v, new_delta_v
