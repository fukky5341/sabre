"""
Note:
- The code assumes the network is a sequence of layers with types defined in LayerType.
- The code supports both fully connected and convolutional layers, as well as ReLU activations.
- Extensive debugging and logging is included for development and verification purposes.
"""

import torch
import gc
import torch.nn.functional as F
from common.network import LayerType
from util.util import compute_input_shapes
from relu.relu_transformer import ReLUTransformer
from common.dataset import Dataset


class BackPropStructRelational:
    """
    Data structure for storing and updating the symbolic coefficients and biases for the lower and upper bounds of the difference (delta) and the two individual inputs as they propagate through the network.

    d = C_d@d + C_d_x1@x1 + C_d_x2@x2 + b
    x1 = C_x1@x1 + b
    x2 = C_x2@x2 + b

    Out <-- vector
    C <-- coefficient matrix
    b <-- bias vector
    d <-- delta
    x1, x2 <-- different inputs

    Attributes:
        d_C_lb (torch.Tensor): Lower bound coefficient matrix for delta.
        d_b_lb (torch.Tensor): Lower bound bias vector for delta.
        d_C_ub (torch.Tensor): Upper bound coefficient matrix for delta.
        d_b_ub (torch.Tensor): Upper bound bias vector for delta.
        d_x1_C_lb (torch.Tensor): Lower bound coefficient matrix for x1.
        d_x1_C_ub (torch.Tensor): Upper bound coefficient matrix for x1.
        d_x2_C_lb (torch.Tensor): Lower bound coefficient matrix for x2.
        d_x2_C_ub (torch.Tensor): Upper bound coefficient matrix for x2.
        x1_C_lb (torch.Tensor): Lower bound coefficient matrix for x1.
        x1_b_lb (torch.Tensor): Lower bound bias vector for x1.
        x1_C_ub (torch.Tensor): Upper bound coefficient matrix for x1.
        x1_b_ub (torch.Tensor): Upper bound bias vector for x1.
        x2_C_lb (torch.Tensor): Lower bound coefficient matrix for x2.
        x2_b_lb (torch.Tensor): Lower bound bias vector for x2.
        x2_C_ub (torch.Tensor): Upper bound coefficient matrix for x2.
        x2_b_ub (torch.Tensor): Upper bound bias vector for x2.
    """

    def __init__(self):

        # $ delta
        '''
        d = C_d@d + C_x1@x1 + C_x2@x2 + b
        '''
        self.d_C_lb = None
        self.d_b_lb = None
        self.d_C_ub = None
        self.d_b_ub = None
        self.d_x1_C_lb = None
        self.d_x1_C_ub = None
        self.d_x2_C_lb = None
        self.d_x2_C_ub = None

        # $ different inputs; x1 and x2
        self.x1_C_lb = None
        self.x1_b_lb = None
        self.x1_C_ub = None
        self.x1_b_ub = None
        self.x2_C_lb = None
        self.x2_b_lb = None
        self.x2_C_ub = None
        self.x2_b_ub = None

    def populate(self, d_C_lb, d_b_lb, d_C_ub, d_b_ub,
                 d_x1_C_lb, d_x1_C_ub, d_x2_C_lb, d_x2_C_ub,
                 x1_C_lb, x1_b_lb, x1_C_ub, x1_b_ub,
                 x2_C_lb, x2_b_lb, x2_C_ub, x2_b_ub):

        self.d_C_lb = d_C_lb
        self.d_b_lb = d_b_lb
        self.d_C_ub = d_C_ub
        self.d_b_ub = d_b_ub
        self.d_x1_C_lb = d_x1_C_lb
        self.d_x1_C_ub = d_x1_C_ub
        self.d_x2_C_lb = d_x2_C_lb
        self.d_x2_C_ub = d_x2_C_ub

        self.x1_C_lb = x1_C_lb
        self.x1_b_lb = x1_b_lb
        self.x1_C_ub = x1_C_ub
        self.x1_b_ub = x1_b_ub
        self.x2_C_lb = x2_C_lb
        self.x2_b_lb = x2_b_lb
        self.x2_C_ub = x2_C_ub
        self.x2_b_ub = x2_b_ub

    def populate_itne(self, d_C_lb, d_b_lb, d_C_ub, d_b_ub,
                      x1_C_lb, x1_b_lb, x1_C_ub, x1_b_ub,
                      x2_C_lb, x2_b_lb, x2_C_ub, x2_b_ub):

        self.d_C_lb = d_C_lb
        self.d_b_lb = d_b_lb
        self.d_C_ub = d_C_ub
        self.d_b_ub = d_b_ub
        self.x1_C_lb = x1_C_lb
        self.x1_b_lb = x1_b_lb
        self.x1_C_ub = x1_C_ub
        self.x1_b_ub = x1_b_ub
        self.x2_C_lb = x2_C_lb
        self.x2_b_lb = x2_b_lb
        self.x2_C_ub = x2_C_ub
        self.x2_b_ub = x2_b_ub

    def delete_old(self):
        del self.d_C_lb, self.d_b_lb, self.d_C_ub, self.d_b_ub, \
            self.d_x1_C_lb, self.d_x1_C_ub, self.d_x2_C_lb, self.d_x2_C_ub
        del self.x1_C_lb, self.x1_b_lb, self.x1_C_ub, self.x1_b_ub, \
            self.x2_C_lb, self.x2_b_lb, self.x2_C_ub, self.x2_b_ub
        gc.collect()
        torch.cuda.empty_cache()

    def print_Cb(self):
        print(f"d_C_lb: {self.d_C_lb}")
        print(f"d_C_ub: {self.d_C_ub}")
        print(f"d_b_lb: {self.d_b_lb}")
        print(f"d_b_ub: {self.d_b_ub}")
        print(f"d_x1_C_lb: {self.d_x1_C_lb}")
        print(f"d_x1_C_ub: {self.d_x1_C_ub}")
        print(f"d_x2_C_lb: {self.d_x2_C_lb}")
        print(f"d_x2_C_ub: {self.d_x2_C_ub}")
        print(f"x1_C_lb: {self.x1_C_lb}")
        print(f"x1_C_ub: {self.x1_C_ub}")
        print(f"x1_b_lb: {self.x1_b_lb}")
        print(f"x1_b_ub: {self.x1_b_ub}")
        print(f"x2_C_lb: {self.x2_C_lb}")
        print(f"x2_C_ub: {self.x2_C_ub}")
        print(f"x2_b_lb: {self.x2_b_lb}")
        print(f"x2_b_ub: {self.x2_b_ub}")


class IndividualAndRelationalBounds(ReLUTransformer):

    def __init__(self, inp1_prop, inp2_prop, net, dataset, delta_eps=None, device='',
                 refine_bounds_prop=False, log_file='log', backprop_mode="normal", clamp_lb_0=False
                 # , noise_ind = None, monotone = False, monotone_prop = 0, monotone_splits = 1, use_all_layers=False, lightweight_diffpoly=False
                 ) -> None:
        """
        Initializes the IndividualAndRelationalBounds class.
        Args:
            inp1_prop (InputProperty): The input property for the first input, including its input and properties.
            inp2_prop (InputProperty): The input property for the second input, including its input and properties.
            net (torch.nn.Module): The neural network model to analyze, including layers and their types, weight, bias.
            device (str, optional): The device to run the computations on. Defaults to ''.
            delta_eps (float, optional): The epsilon value for delta bounds. Defaults to 0.01.
            refine_bounds_prop (bool, optional): Whether to refine bounds using relational properties. Defaults to False.
            log_file (str, optional): The file path for logging results. Defaults to 'log'.
            lbs, ubs (List[torch.Tensor], List[torch.Tensor]): Lower and upper bounds for the inputs and delta at each layer.
            diff (torch.Tensor): The difference between the two inputs, reshaped to a flat vector.

        """
        self.inp1 = inp1_prop.input
        self.inp2 = inp2_prop.input
        if dataset == Dataset.ACAS:
            self.inp1 = inp1_prop.input_props[0].input_lb
            self.inp2 = inp2_prop.input_props[0].input_lb
        self.inp1_correct_label = inp1_prop.out_constr.label.item()
        self.inp2_correct_label = inp2_prop.out_constr.label.item()
        self.input_shape = self.inp1.shape
        if self.input_shape not in [(1, 28, 28), (3, 32, 32), (1, 1, 2), (1, 1, 12), (1, 1, 87), (5,)]:
            raise ValueError(f"Unrecognised input shape {self.input_shape}")
        self.net = net
        self.pre_IARb = None
        self.inp1_input_lb = inp1_prop.input_props[0].input_lb
        self.inp1_input_ub = inp1_prop.input_props[0].input_ub
        self.inp2_input_lb = inp2_prop.input_props[0].input_lb
        self.inp2_input_ub = inp2_prop.input_props[0].input_ub
        self.inp1_lbs = None
        self.inp1_ubs = None
        self.inp2_lbs = None
        self.inp2_ubs = None
        self.d_lbs = None
        self.d_ubs = None
        self.inp1_relu_input_info = None
        self.inp2_relu_input_info = None
        self.shapes = compute_input_shapes(net=self.net, input_shape=self.input_shape)
        self.diff = None
        if delta_eps is None:
            self.diff = self.inp1.reshape(-1) - self.inp2.reshape(-1)
        self.delta_eps = delta_eps
        self.linear_conv_layer_indices = []
        self.device = device
        self.refine_bounds_prop = refine_bounds_prop
        self.log_file = log_file
        self.clamp_lb_0 = clamp_lb_0
        self.backprop_mode = backprop_mode
        self.feasible_flag = True
        if self.backprop_mode == "normal":
            self.handle_relu = self.handle_relu_normal
        elif self.backprop_mode == "DP":
            self.handle_relu = self.handle_relu_DP

    def handle_linear_IAR(self, linear_wt, bias, back_prop_struct):
        """
        Handles the symbolic propagation at the linear layer.
        Args:
            linear_wt (torch.Tensor): The weight matrix of the linear layer.
            bias (torch.Tensor): The bias vector of the linear layer.
            back_prop_struct (BackPropStructRelational): The structure holding the symbolic coefficients and biases.

        Symbolic propagation for the linear layer is defined as:
            for individual inputs x1 and x2:
            x1 = W_x1@x1 + b_x1 --substitute--> (propagated form) x1 = C_x1@'x1' + 'b_x1' = C_x1@(W_x1@x1 + b_x1) + 'b_x1'
            x2 = W_x2@x2 + b_x2 --substitute--> (propagated form) x2 = C_x2@'x2' + 'b_x2' = C_x2@(W_x2@x2 + b_x2) + 'b_x2'
            for delta:
            d = W_d@d + b_d --substitute--> (propagated form) d = C_d@'d' + C_d_x1@'x1' + C_d_x2@'x2' + 'b_d' = C_d@(W_d@d + b_d) + C_d_x1@(W_x1@x1 + b_x1) + C_d_x2@(W_x2@x2 + b_x2) + 'b_d'
        """

        # $ for single
        x1_b_lb = back_prop_struct.x1_b_lb + back_prop_struct.x1_C_lb.matmul(bias)
        x2_b_lb = back_prop_struct.x2_b_lb + back_prop_struct.x2_C_lb.matmul(bias)
        x1_b_ub = back_prop_struct.x1_b_ub + back_prop_struct.x1_C_ub.matmul(bias)
        x2_b_ub = back_prop_struct.x2_b_ub + back_prop_struct.x2_C_ub.matmul(bias)

        x1_C_lb = back_prop_struct.x1_C_lb.matmul(linear_wt)
        x2_C_lb = back_prop_struct.x2_C_lb.matmul(linear_wt)
        x1_C_ub = back_prop_struct.x1_C_ub.matmul(linear_wt)
        x2_C_ub = back_prop_struct.x2_C_ub.matmul(linear_wt)

        # $ for delta
        if back_prop_struct.d_x1_C_lb is not None:
            d_b_lb = back_prop_struct.d_b_lb + back_prop_struct.d_x1_C_lb.matmul(bias)
            d_b_lb = d_b_lb + back_prop_struct.d_x2_C_lb.matmul(bias)
            d_b_ub = back_prop_struct.d_b_ub + back_prop_struct.d_x1_C_ub.matmul(bias)
            d_b_ub = d_b_ub + back_prop_struct.d_x2_C_ub.matmul(bias)
        else:
            d_b_lb = back_prop_struct.d_b_lb
            d_b_ub = back_prop_struct.d_b_ub

        d_C_lb = back_prop_struct.d_C_lb.matmul(linear_wt)
        d_C_ub = back_prop_struct.d_C_ub.matmul(linear_wt)
        if back_prop_struct.d_x1_C_lb is not None:
            d_x1_C_lb = back_prop_struct.d_x1_C_lb.matmul(linear_wt)
            d_x1_C_ub = back_prop_struct.d_x1_C_ub.matmul(linear_wt)
            d_x2_C_lb = back_prop_struct.d_x2_C_lb.matmul(linear_wt)
            d_x2_C_ub = back_prop_struct.d_x2_C_ub.matmul(linear_wt)
        else:
            d_x1_C_lb = None
            d_x1_C_ub = None
            d_x2_C_lb = None
            d_x2_C_ub = None

        back_prop_struct.populate(d_C_lb=d_C_lb, d_b_lb=d_b_lb,
                                  d_C_ub=d_C_ub, d_b_ub=d_b_ub,
                                  d_x1_C_lb=d_x1_C_lb,
                                  d_x1_C_ub=d_x1_C_ub,
                                  d_x2_C_lb=d_x2_C_lb,
                                  d_x2_C_ub=d_x2_C_ub,
                                  x1_C_lb=x1_C_lb, x2_C_lb=x2_C_lb,
                                  x1_C_ub=x1_C_ub, x2_C_ub=x2_C_ub,
                                  x1_b_lb=x1_b_lb, x2_b_lb=x2_b_lb,
                                  x1_b_ub=x1_b_ub, x2_b_ub=x2_b_ub)

        return back_prop_struct

    # Transformer for convolutional linear/affine layers.

    def handle_conv_IAR(self, conv_weight, conv_bias, back_prop_struct, preconv_shape, postconv_shape,
                        stride, padding, groups=1, dilation=(1, 1)):
        """
        Handles the symbolic propagation at the convolutional layer.
        Args:
            conv_weight (torch.Tensor): The weight tensor of the convolutional layer.
            conv_bias (torch.Tensor): The bias tensor of the convolutional layer.
            back_prop_struct (BackPropStructRelational): The structure holding the symbolic coefficients and biases.
            preconv_shape (tuple): The shape of the input tensor before convolution.
            postconv_shape (tuple): The shape of the output tensor after convolution.
            stride (tuple): The stride of the convolution.
            padding (tuple): The padding applied to the input tensor.
            groups (int, optional): The number of groups for the convolution. Defaults to 1.
            dilation (tuple, optional): The dilation rate for the convolution. Defaults to (1, 1).

        Symbolic propagation for the convolutional layer is defined similarly to the linear layer,
        but takes into account the spatial dimensions and the convolution operation.
        """

        kernel_hw = conv_weight.shape[-2:]
        h_padding = (preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)) % stride[0]
        w_padding = (preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = back_prop_struct.d_C_lb.shape
        d_C_lb = back_prop_struct.d_C_lb.view((coef_shape[0], *postconv_shape))
        d_C_ub = back_prop_struct.d_C_ub.view((coef_shape[0], *postconv_shape))

        if back_prop_struct.d_x1_C_lb is not None:
            d_x1_C_lb = back_prop_struct.d_x1_C_lb.view((coef_shape[0], *postconv_shape))
            d_x1_C_ub = back_prop_struct.d_x1_C_ub.view((coef_shape[0], *postconv_shape))
            d_x2_C_lb = back_prop_struct.d_x2_C_lb.view((coef_shape[0], *postconv_shape))
            d_x2_C_ub = back_prop_struct.d_x2_C_ub.view((coef_shape[0], *postconv_shape))

        if back_prop_struct.x1_C_lb is not None:
            x1_C_lb = back_prop_struct.x1_C_lb.view((coef_shape[0], *postconv_shape))
            x2_C_lb = back_prop_struct.x2_C_lb.view((coef_shape[0], *postconv_shape))
            x1_C_ub = back_prop_struct.x1_C_ub.view((coef_shape[0], *postconv_shape))
            x2_C_ub = back_prop_struct.x2_C_ub.view((coef_shape[0], *postconv_shape))

        if back_prop_struct.d_x1_C_lb is not None:
            d_b_lb = back_prop_struct.d_b_lb + (d_x1_C_lb.sum((2, 3)) * conv_bias).sum(1)
            d_b_lb = d_b_lb + (d_x2_C_lb.sum((2, 3)) * conv_bias).sum(1)
            d_b_ub = back_prop_struct.d_b_ub + (d_x1_C_ub.sum((2, 3)) * conv_bias).sum(1)
            d_b_ub = d_b_ub + (d_x2_C_ub.sum((2, 3)) * conv_bias).sum(1)
        else:
            d_b_lb = back_prop_struct.d_b_lb
            d_b_ub = back_prop_struct.d_b_ub

        if back_prop_struct.x1_C_lb is not None:
            x1_b_lb = back_prop_struct.x1_b_lb + (x1_C_lb.sum((2, 3)) * conv_bias).sum(1)
            x2_b_lb = back_prop_struct.x2_b_lb + (x2_C_lb.sum((2, 3)) * conv_bias).sum(1)
            x1_b_ub = back_prop_struct.x1_b_ub + (x1_C_ub.sum((2, 3)) * conv_bias).sum(1)
            x2_b_ub = back_prop_struct.x2_b_ub + (x2_C_ub.sum((2, 3)) * conv_bias).sum(1)

        new_d_C_lb = F.conv_transpose2d(d_C_lb, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
        new_d_C_ub = F.conv_transpose2d(d_C_ub, conv_weight, None, stride, padding,
                                        output_padding, groups, dilation)
        if back_prop_struct.d_x1_C_lb is not None:
            new_d_n1_C_lb = F.conv_transpose2d(d_x1_C_lb, conv_weight, None, stride, padding,
                                               output_padding, groups, dilation)
            new_d_n1_C_ub = F.conv_transpose2d(d_x1_C_ub, conv_weight, None, stride, padding,
                                               output_padding, groups, dilation)
            new_d_n2_C_lb = F.conv_transpose2d(d_x2_C_lb, conv_weight, None, stride, padding,
                                               output_padding, groups, dilation)
            new_d_n2_C_ub = F.conv_transpose2d(d_x2_C_ub, conv_weight, None, stride, padding,
                                               output_padding, groups, dilation)
        else:
            new_d_n1_C_lb = None
            new_d_n1_C_ub = None
            new_d_n2_C_lb = None
            new_d_n2_C_ub = None

        if back_prop_struct.x1_C_lb is not None:
            x1_C_lb = F.conv_transpose2d(x1_C_lb, conv_weight, None, stride, padding,
                                         output_padding, groups, dilation)
            x2_C_lb = F.conv_transpose2d(x2_C_lb, conv_weight, None, stride, padding,
                                         output_padding, groups, dilation)
            x1_C_ub = F.conv_transpose2d(x1_C_ub, conv_weight, None, stride, padding,
                                         output_padding, groups, dilation)
            x2_C_ub = F.conv_transpose2d(x2_C_ub, conv_weight, None, stride, padding,
                                         output_padding, groups, dilation)

        new_d_C_lb = new_d_C_lb.view((coef_shape[0], -1))
        new_d_C_ub = new_d_C_ub.view((coef_shape[0], -1))

        if new_d_n1_C_lb is not None:
            new_d_n1_C_lb = new_d_n1_C_lb.view((coef_shape[0], -1))
            new_d_n1_C_ub = new_d_n1_C_ub.view((coef_shape[0], -1))
            new_d_n2_C_lb = new_d_n2_C_lb.view((coef_shape[0], -1))
            new_d_n2_C_ub = new_d_n2_C_ub.view((coef_shape[0], -1))

        if back_prop_struct.x1_C_lb is not None:
            x1_C_lb = x1_C_lb.view((coef_shape[0], -1))
            x2_C_lb = x2_C_lb.view((coef_shape[0], -1))
            x1_C_ub = x1_C_ub.view((coef_shape[0], -1))
            x2_C_ub = x2_C_ub.view((coef_shape[0], -1))
        else:
            x1_C_lb = None
            x2_C_lb = None
            x1_C_ub = None
            x2_C_ub = None
            x1_b_lb = None
            x2_b_lb = None
            x1_b_ub = None
            x2_b_ub = None

        back_prop_struct.populate(d_C_lb=new_d_C_lb, d_b_lb=d_b_lb,
                                  d_C_ub=new_d_C_ub, d_b_ub=d_b_ub,
                                  d_x1_C_lb=new_d_n1_C_lb,
                                  d_x1_C_ub=new_d_n1_C_ub,
                                  d_x2_C_lb=new_d_n2_C_lb,
                                  d_x2_C_ub=new_d_n2_C_ub,
                                  x1_C_lb=x1_C_lb, x2_C_lb=x2_C_lb,
                                  x1_C_ub=x1_C_ub, x2_C_ub=x2_C_ub,
                                  x1_b_lb=x1_b_lb, x2_b_lb=x2_b_lb,
                                  x1_b_ub=x1_b_ub, x2_b_ub=x2_b_ub)
        return back_prop_struct

    def pos_neg_weight_decomposition(self, coef):
        """
        Decomposes the coefficient tensor into positive and negative components.
        """
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device=self.device))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device=self.device))
        return neg_comp, pos_comp

    # note: handling relu is done in the ReLUTransformer class.

    def handle_layer_IAR(self, prop_struct, layer, linear_layer_idx, lubs_layer_idx):
        """
        Handles the layer-wise symbolic propagation for the given layer.
        Args:
            prop_struct (BackPropStructRelational): The structure holding the symbolic coefficients and biases.
            layer (Layer): The current layer in the neural network (e.g., Linear, Conv2D, ReLU).
            linear_layer_idx (int): The index of the current linear layer in the network.
            lubs_layer_idx (int): The index of the current layer for lower and upper bounds.
                e.g., currently propagating symbolic bounds for i-th relu layer, then lubs_layer_idx = i indicates the i-th lbs and ubs in the stored lists for getting the relu conditions
                note: lbs and ubs lists include the input layer, but net does not.
        Returns:
            BackPropStructRelational: The updated structure with propagated symbolic coefficients and biases.
        """
        if layer.type is LayerType.Linear:
            back_prop_struct = self.handle_linear_IAR(linear_wt=layer.weight,
                                                      bias=layer.bias, back_prop_struct=prop_struct)
        elif layer.type is LayerType.Conv2D:
            back_prop_struct = self.handle_conv_IAR(conv_weight=layer.weight, conv_bias=layer.bias,
                                                    back_prop_struct=prop_struct,
                                                    preconv_shape=self.shapes[linear_layer_idx], postconv_shape=self.shapes[linear_layer_idx + 1],
                                                    stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
        elif layer.type is LayerType.ReLU:
            back_prop_struct = self.handle_relu(
                back_prop_struct=prop_struct,
                relu_input_layer_idx=lubs_layer_idx,
                inp1_lb_layer=self.inp1_lbs[lubs_layer_idx],
                inp1_ub_layer=self.inp1_ubs[lubs_layer_idx],
                inp2_lb_layer=self.inp2_lbs[lubs_layer_idx],
                inp2_ub_layer=self.inp2_ubs[lubs_layer_idx],
                d_lb_layer=self.d_lbs[lubs_layer_idx],
                d_ub_layer=self.d_ubs[lubs_layer_idx])
        else:
            raise NotImplementedError(f'diff verifier for {layer.type} is not implemented')
        return back_prop_struct

    def refine_bounds_IAR(self, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub, layer_idx=None, ):
        """
        refine delta based on d = n1 - n2
        """
        r_d_lb = torch.max(d_lb, inp1_lb - inp2_ub)
        r_d_ub = torch.min(d_ub, inp1_ub - inp2_lb)

        """
        refine individual bounds based on delta bounds
        n1 = n2 + d
        n2 = n1 - d
        """
        r_inp1_lb = torch.max(inp1_lb, inp2_lb + d_lb)
        r_inp1_ub = torch.min(inp1_ub, inp2_ub + d_ub)
        r_inp2_lb = torch.max(inp2_lb, inp1_lb - d_ub)
        r_inp2_ub = torch.min(inp2_ub, inp1_ub - d_lb)
        r_d_lb = torch.max(r_d_lb, r_inp1_lb - r_inp2_ub)
        r_d_ub = torch.min(r_d_ub, r_inp1_ub - r_inp2_lb)

        return r_inp1_lb, r_inp1_ub, r_inp2_lb, r_inp2_ub, r_d_lb, r_d_ub

    # Helper function for computing concrete bounds from symbolic bounds.
    def concretize_bounds(self, back_prop_struct, d_lb_layer, d_ub_layer,
                          inp1_lb_layer, inp1_ub_layer, inp2_lb_layer, inp2_ub_layer, layer_idx, lubs_layer_idx=None):
        """
        Computes the concrete bounds for the given layer based on the symbolic bounds, while taking the worst-case scenario into account.
        Args:
            back_prop_struct (BackPropStructRelational): The structure holding the symbolic coefficients and biases.
            d_lb_layer (torch.Tensor): Lower bound for delta at the current layer.
            d_ub_layer (torch.Tensor): Upper bound for delta at the current layer.
            inp1_lb_layer (torch.Tensor): Lower bound for input 1 at the current layer.
            inp1_ub_layer (torch.Tensor): Upper bound for input 1 at the current layer.
            inp2_lb_layer (torch.Tensor): Lower bound for input 2 at the current layer.
            inp2_ub_layer (torch.Tensor): Upper bound for input 2 at the current layer.
            layer_idx (int, optional): The index of the current layer. Defaults to None.
        In the worst-case scenario, the bounds are computed as follows:
            For lower bounds; lb = neg_comp_lb @ ub + pos_comp_lb @ lb + b_lb
            For upper bounds; ub = neg_comp_ub @ lb + pos_comp_ub @ ub + b_ub
        General form for each input and delta:
            d = C_d@d + C_d_x1@x1 + C_d_x2@x2 + b
            x1 = C_x1@x1 + b
            x2 = C_x2@x2 + b
        Note:
            @ and matmul denote matrix product
            * denotes element-wise product
        """
        # $ for single -->
        neg_comp_lb_inp1, pos_comp_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.x1_C_lb)
        neg_comp_ub_inp1, pos_comp_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.x1_C_ub)
        neg_comp_lb_inp2, pos_comp_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.x2_C_lb)
        neg_comp_ub_inp2, pos_comp_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.x2_C_ub)

        lb_inp1 = neg_comp_lb_inp1 @ inp1_ub_layer + pos_comp_lb_inp1 @ inp1_lb_layer + back_prop_struct.x1_b_lb
        ub_inp1 = neg_comp_ub_inp1 @ inp1_lb_layer + pos_comp_ub_inp1 @ inp1_ub_layer + back_prop_struct.x1_b_ub
        lb_inp2 = neg_comp_lb_inp2 @ inp2_ub_layer + pos_comp_lb_inp2 @ inp2_lb_layer + back_prop_struct.x2_b_lb
        ub_inp2 = neg_comp_ub_inp2 @ inp2_lb_layer + pos_comp_ub_inp2 @ inp2_ub_layer + back_prop_struct.x2_b_ub

        # $ <-- for single

        # $ for relational -->
        neg_comp_d_lb, pos_comp_d_lb = self.pos_neg_weight_decomposition(back_prop_struct.d_C_lb)
        neg_comp_d_ub, pos_comp_d_ub = self.pos_neg_weight_decomposition(back_prop_struct.d_C_ub)
        if back_prop_struct.d_x1_C_lb is not None:
            neg_comp_d_lb_inp1, pos_comp_d_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_lb)
            neg_comp_d_ub_inp1, pos_comp_d_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_ub)
            neg_comp_d_lb_inp2, pos_comp_d_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_lb)
            neg_comp_d_ub_inp2, pos_comp_d_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_ub)

        # print(f"neg_comp_d_lb: {neg_comp_d_lb}, d_ub_layer: {d_ub_layer}")
        lb = neg_comp_d_lb @ d_ub_layer + pos_comp_d_lb @ d_lb_layer + back_prop_struct.d_b_lb
        if back_prop_struct.d_x1_C_lb is not None:
            lb = lb + neg_comp_d_lb_inp1 @ inp1_ub_layer + pos_comp_d_lb_inp1 @ inp1_lb_layer
            lb = lb + neg_comp_d_lb_inp2 @ inp2_ub_layer + pos_comp_d_lb_inp2 @ inp2_lb_layer

        ub = neg_comp_d_ub @ d_lb_layer + pos_comp_d_ub @ d_ub_layer + back_prop_struct.d_b_ub
        if back_prop_struct.d_x1_C_lb is not None:
            ub = ub + neg_comp_d_ub_inp1 @ inp1_lb_layer + pos_comp_d_ub_inp1 @ inp1_ub_layer
            ub = ub + neg_comp_d_ub_inp2 @ inp2_lb_layer + pos_comp_d_ub_inp2 @ inp2_ub_layer

        # $ <-- for relational

        return lb_inp1, ub_inp1, lb_inp2, ub_inp2, lb, ub

    def concretize_delta_bounds(self, back_prop_struct, d_lb_layer, d_ub_layer,
                                inp1_lb_layer, inp1_ub_layer, inp2_lb_layer, inp2_ub_layer, layer_idx, lubs_layer_idx=None):

        # $ for relational -->
        neg_comp_d_lb, pos_comp_d_lb = self.pos_neg_weight_decomposition(back_prop_struct.d_C_lb)
        neg_comp_d_ub, pos_comp_d_ub = self.pos_neg_weight_decomposition(back_prop_struct.d_C_ub)
        if back_prop_struct.d_x1_C_lb is not None:
            neg_comp_d_lb_inp1, pos_comp_d_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_lb)
            neg_comp_d_ub_inp1, pos_comp_d_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_ub)
            neg_comp_d_lb_inp2, pos_comp_d_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_lb)
            neg_comp_d_ub_inp2, pos_comp_d_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_ub)

        lb = neg_comp_d_lb @ d_ub_layer + pos_comp_d_lb @ d_lb_layer + back_prop_struct.d_b_lb
        if back_prop_struct.d_x1_C_lb is not None:
            lb = lb + neg_comp_d_lb_inp1 @ inp1_ub_layer + pos_comp_d_lb_inp1 @ inp1_lb_layer
            lb = lb + neg_comp_d_lb_inp2 @ inp2_ub_layer + pos_comp_d_lb_inp2 @ inp2_lb_layer

        ub = neg_comp_d_ub @ d_lb_layer + pos_comp_d_ub @ d_ub_layer + back_prop_struct.d_b_ub
        if back_prop_struct.d_x1_C_lb is not None:
            ub = ub + neg_comp_d_ub_inp1 @ inp1_lb_layer + pos_comp_d_ub_inp1 @ inp1_ub_layer
            ub = ub + neg_comp_d_ub_inp2 @ inp2_lb_layer + pos_comp_d_ub_inp2 @ inp2_ub_layer
        # $ <-- for relational

        return lb, ub

    # Debugging for avoid errors due to floating point imprecision.
    def check_lb_ub_correctness(self, lb, ub, layer_idx=None):
        if not torch.all(lb <= ub + 1e-6):
            # debug
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\nDetected inconsistencies in bounds for layer {layer_idx}\n")
                f.write(f"(lb, ub, ub - lb)\n")
                for i in range(len(lb)):
                    if ub[i].item() < lb[i].item():
                        f.write(f"{i}: ({lb[i].item()}, {ub[i].item()}, {ub[i].item() - lb[i].item()})\n")
            # assert torch.all(lb <= ub + 1e-6)
            return True
        return False

    def get_layer_size(self, linear_layer_index):
        layer = self.net[self.linear_conv_layer_indices[linear_layer_index]]
        if layer.type is LayerType.Linear:
            shape = self.shapes[linear_layer_index + 1]
            return shape
        if layer.type is LayerType.Conv2D:
            shape = self.shapes[linear_layer_index + 1]
            return (shape[0] * shape[1] * shape[2])

    def initialize_back_prop_struct(self, layer_idx):
        layer_size = self.get_layer_size(linear_layer_index=layer_idx)
        d_C_lb = torch.eye(n=layer_size, device=self.device)
        d_b_lb = torch.zeros(layer_size, device=self.device)
        d_C_ub = torch.eye(n=layer_size, device=self.device)
        d_b_ub = torch.zeros(layer_size, device=self.device)
        d_x1_C_lb = torch.zeros((layer_size, layer_size), device=self.device)
        d_x1_C_ub = torch.zeros((layer_size, layer_size), device=self.device)
        d_x2_C_lb = torch.zeros((layer_size, layer_size), device=self.device)
        d_x2_C_ub = torch.zeros((layer_size, layer_size), device=self.device)

        x1_C_lb = torch.eye(n=layer_size, device=self.device)
        x2_C_lb = torch.eye(n=layer_size, device=self.device)
        x1_C_ub = torch.eye(n=layer_size, device=self.device)
        x2_C_ub = torch.eye(n=layer_size, device=self.device)

        x1_b_lb = torch.zeros(layer_size, device=self.device)
        x2_b_lb = torch.zeros(layer_size, device=self.device)
        x1_b_ub = torch.zeros(layer_size, device=self.device)
        x2_b_ub = torch.zeros(layer_size, device=self.device)

        back_prop_struct = BackPropStructRelational()
        back_prop_struct.populate(d_C_lb=d_C_lb, d_b_lb=d_b_lb,
                                  d_C_ub=d_C_ub, d_b_ub=d_b_ub,
                                  d_x1_C_lb=d_x1_C_lb,
                                  d_x1_C_ub=d_x1_C_ub,
                                  d_x2_C_lb=d_x2_C_lb,
                                  d_x2_C_ub=d_x2_C_ub,
                                  x1_C_lb=x1_C_lb, x2_C_lb=x2_C_lb,
                                  x1_C_ub=x1_C_ub, x2_C_ub=x2_C_ub,
                                  x1_b_lb=x1_b_lb, x2_b_lb=x2_b_lb,
                                  x1_b_ub=x1_b_ub, x2_b_ub=x2_b_ub)
        return back_prop_struct

    def back_substitution_IAR(self, layer_idx, pre_IARb=None):
        # def back_substitution_IAR(self, layer_idx, d_lbs, d_ubs):
        """
        Back substitution for Individual And Relational bounds.
        This function computes the concrete bounds at the net[layer_idx] layer
        """

        if layer_idx != len(self.d_lbs) - 1 or layer_idx != len(self.d_ubs) - 1:
            # note: layer_idx is the index of the layer in the net, but d_lbs and d_ubs are lists that include input layer
            raise ValueError(f"Size of lower bounds computed in previous layers don't match; layer_idx: {layer_idx}, len(d_lbs): {len(self.d_lbs)}")

        back_prop_struct = None
        inp1_lb = None
        inp1_ub = None
        inp2_lb = None
        inp2_ub = None
        d_lb = None
        d_ub = None
        # Assuming the network has alternate affine and activation layer.
        linear_layer_index = layer_idx // 2  # e.g., 0 //2 = 0, 1 // 2 = 0, 2 // 2 = 1, 3 // 2 = 1, etc.

        # * if layer_idx = 2, then r_layer_idx = 2, 1, 0
        for r_layer_idx in reversed(range(layer_idx + 1)):

            if back_prop_struct is None:
                back_prop_struct = self.initialize_back_prop_struct(layer_idx=linear_layer_index)
            curr_layer = self.net[r_layer_idx]
            back_prop_struct = self.handle_layer_IAR(prop_struct=back_prop_struct, layer=curr_layer, linear_layer_idx=linear_layer_index,
                                                     lubs_layer_idx=r_layer_idx)
            # ** for "lubs_layer_idx=r_layer_idx", this is required for relu conditions **
            # ** lbs[r_layer_idx] gives the lower bounds of previous layers' variables **

            if back_prop_struct is not None:
                if r_layer_idx == 0:
                    new_inp1_lb, new_inp1_ub, new_inp2_lb, new_inp2_ub, new_d_lb, new_d_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,
                                                                                                                    d_lb_layer=self.d_lbs[r_layer_idx],
                                                                                                                    d_ub_layer=self.d_ubs[r_layer_idx],
                                                                                                                    inp1_lb_layer=self.inp1_lbs[r_layer_idx],
                                                                                                                    inp1_ub_layer=self.inp1_ubs[r_layer_idx],
                                                                                                                    inp2_lb_layer=self.inp2_lbs[r_layer_idx],
                                                                                                                    inp2_ub_layer=self.inp2_ubs[r_layer_idx],
                                                                                                                    layer_idx=layer_idx, lubs_layer_idx=r_layer_idx)

                    inp1_lb = (new_inp1_lb if inp1_lb is None else (torch.max(inp1_lb, new_inp1_lb)))
                    inp1_ub = (new_inp1_ub if inp1_ub is None else (torch.min(inp1_ub, new_inp1_ub)))
                    inp2_lb = (new_inp2_lb if inp2_lb is None else (torch.max(inp2_lb, new_inp2_lb)))
                    inp2_ub = (new_inp2_ub if inp2_ub is None else (torch.min(inp2_ub, new_inp2_ub)))
                    d_lb = (new_d_lb if d_lb is None else (torch.max(d_lb, new_d_lb)))
                    d_ub = (new_d_ub if d_ub is None else (torch.min(d_ub, new_d_ub)))
                    inp1_lbub_alarm = self.check_lb_ub_correctness(lb=new_inp1_lb, ub=new_inp1_ub, layer_idx=layer_idx)
                    inp2_lbub_alarm = self.check_lb_ub_correctness(lb=new_inp2_lb, ub=new_inp2_ub, layer_idx=layer_idx)
                    d_lbub_alarm = self.check_lb_ub_correctness(lb=new_d_lb, ub=new_d_ub, layer_idx=layer_idx)
                    # flag: true if any lb > ub
                    if inp1_lbub_alarm or inp2_lbub_alarm or d_lbub_alarm:
                        return None, None, None, None, None, None
                else:
                    new_d_lb, new_d_ub = self.concretize_delta_bounds(back_prop_struct=back_prop_struct,
                                                                      d_lb_layer=self.d_lbs[r_layer_idx],
                                                                      d_ub_layer=self.d_ubs[r_layer_idx],
                                                                      inp1_lb_layer=self.inp1_lbs[r_layer_idx],
                                                                      inp1_ub_layer=self.inp1_ubs[r_layer_idx],
                                                                      inp2_lb_layer=self.inp2_lbs[r_layer_idx],
                                                                      inp2_ub_layer=self.inp2_ubs[r_layer_idx],
                                                                      layer_idx=layer_idx, lubs_layer_idx=r_layer_idx)
                    d_lb = (new_d_lb if d_lb is None else (torch.max(d_lb, new_d_lb)))
                    d_ub = (new_d_ub if d_ub is None else (torch.min(d_ub, new_d_ub)))
                    d_lbub_alarm = self.check_lb_ub_correctness(lb=new_d_lb, ub=new_d_ub, layer_idx=layer_idx)
                    # flag: true if any lb > ub
                    if d_lbub_alarm:
                        return None, None, None, None, None, None

            if curr_layer.type in [LayerType.Linear, LayerType.Conv2D]:
                linear_layer_index -= 1

        if self.clamp_lb_0:
            if self.net[layer_idx].type is LayerType.ReLU:
                inp1_lb = torch.clamp(inp1_lb, min=0)  # e.g., inp1_lb = torch.max(inp1_lb, torch.zeros_like(inp1_lb, device=self.device))
                inp2_lb = torch.clamp(inp2_lb, min=0)

        if self.refine_bounds_prop:
            inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub = self.refine_bounds_IAR(inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub, layer_idx=layer_idx)
            inp1_lbub_alarm = self.check_lb_ub_correctness(lb=inp1_lb, ub=inp1_ub, layer_idx=layer_idx)
            inp2_lbub_alarm = self.check_lb_ub_correctness(lb=inp2_lb, ub=inp2_ub, layer_idx=layer_idx)
            d_lbub_alarm = self.check_lb_ub_correctness(lb=d_lb, ub=d_ub, layer_idx=layer_idx)
            if inp1_lbub_alarm or inp2_lbub_alarm or d_lbub_alarm:
                return None, None, None, None, None, None

        back_prop_struct.delete_old()
        return inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub

    def update_bounds_IAR(self, ds_history=None, ns_history=None, split_layer_list=None):
        """
        1. replace the preactivation bounds from ds_history
        2. recompute the subsequent layers' bounds
        3. update self.inp1_lbs, self.inp1_ubs, self.inp2_lbs, self.inp2_ubs, self.d_lbs, self.d_ubs
        4. return nothing

        split_layer_list: list of lubs layer indices where the splits happen
        start_layer_idx indicates the lubs layer index of preactivation and also the relu layer index of net
        """
        start_layer_idx = min(split_layer_list)
        # truncate the bounds lists
        self.inp1_lbs = self.inp1_lbs[:start_layer_idx + 1]  # e.g., if start_layer_idx = 2, keep lubs[0], lubs[1], lubs[2]
        self.inp1_ubs = self.inp1_ubs[:start_layer_idx + 1]
        self.inp2_lbs = self.inp2_lbs[:start_layer_idx + 1]
        self.inp2_ubs = self.inp2_ubs[:start_layer_idx + 1]
        self.d_lbs = self.d_lbs[:start_layer_idx + 1]
        self.d_ubs = self.d_ubs[:start_layer_idx + 1]
        for layer_idx in range(start_layer_idx, len(self.net)):
            if self.net[layer_idx].type == LayerType.ReLU and layer_idx in split_layer_list:  # update bounds of preactivation layer
                if ds_history is not None:
                    temp_ds_info_list = [ds_info for ds_info in ds_history if ds_info.layer_idx == layer_idx]
                    for ds_info in temp_ds_info_list:
                        inp1_lb = self.inp1_lbs[layer_idx][ds_info.pos]
                        inp1_ub = self.inp1_ubs[layer_idx][ds_info.pos]
                        inp2_lb = self.inp2_lbs[layer_idx][ds_info.pos]
                        inp2_ub = self.inp2_ubs[layer_idx][ds_info.pos]
                        d_lb = self.d_lbs[layer_idx][ds_info.pos]
                        d_ub = self.d_ubs[layer_idx][ds_info.pos]

                        if '1' in ds_info.ds_type:
                            # update bounds
                            new_d_lb = d_lb
                            new_d_ub = ds_info.split_value
                            new_inp1_lb = inp1_lb  # lb1 = lb2 + d_lb
                            new_inp2_ub = inp2_ub  # ub2 = ub1 - d_lb
                            new_inp1_ub = min(inp1_ub, inp2_ub + new_d_ub)  # ub1 = ub2 + new_d_ub
                            new_inp2_lb = max(inp2_lb, inp1_lb - new_d_ub)  # lb2 = lb1 - new_d_ub
                        elif '2' in ds_info.ds_type:
                            # update bounds
                            new_d_lb = ds_info.split_value
                            new_d_ub = d_ub
                            new_inp1_ub = inp1_ub  # ub1 = ub2 + d_ub
                            new_inp2_lb = inp2_lb  # lb2 = lb1 - d_ub
                            new_inp1_lb = max(inp1_lb, inp2_lb + new_d_lb)  # lb1 = lb2 + new_d_lb
                            new_inp2_ub = min(inp2_ub, inp1_ub - new_d_lb)  # ub2 = ub1 - new_d_lb
                        else:
                            raise ValueError("Unknown ds_type")
                    self.inp1_lbs[layer_idx][ds_info.pos] = new_inp1_lb
                    self.inp1_ubs[layer_idx][ds_info.pos] = new_inp1_ub
                    self.inp2_lbs[layer_idx][ds_info.pos] = new_inp2_lb
                    self.inp2_ubs[layer_idx][ds_info.pos] = new_inp2_ub
                    self.d_lbs[layer_idx][ds_info.pos] = new_d_lb
                    self.d_ubs[layer_idx][ds_info.pos] = new_d_ub
                elif ns_history is not None:
                    temp_ns_info_list = [ns_info for ns_info in ns_history if ns_info.layer_idx == layer_idx]
                    for ns_info in temp_ns_info_list:
                        inp1_lb = self.inp1_lbs[layer_idx][ns_info.pos]
                        inp1_ub = self.inp1_ubs[layer_idx][ns_info.pos]
                        inp2_lb = self.inp2_lbs[layer_idx][ns_info.pos]
                        inp2_ub = self.inp2_ubs[layer_idx][ns_info.pos]
                        d_lb = self.d_lbs[layer_idx][ns_info.pos]
                        d_ub = self.d_ubs[layer_idx][ns_info.pos]

                        if 'A1' in ns_info.ns_type:
                            # update bounds
                            new_inp1_lb = inp1_lb  # lb1 = lb2 + d_lb
                            new_inp2_lb = inp2_lb  # lb2 = lb1 - d_ub
                            new_d_lb = d_lb  # d_lb = lb1 - ub2
                            new_inp1_ub = 0  # ub1 = 0
                            new_inp2_ub = min(inp2_ub, -d_lb)  # ub2 = ub1 - d_lb
                            new_d_ub = min(d_ub, -inp2_lb)  # d_ub = ub1 - lb2
                        elif 'A2' in ns_info.ns_type:
                            # update bounds
                            new_inp1_ub = inp1_ub  # ub1 = ub2 + d_ub
                            new_inp2_ub = inp2_ub  # ub2 = ub1 - d_lb
                            new_d_ub = d_ub  # d_ub = ub1 - lb2
                            new_inp1_lb = 0  # lb1 = 0
                            new_inp2_lb = max(inp2_lb, -d_ub)  # lb2 = lb1 - d_ub
                            new_d_lb = max(d_lb, -inp2_ub)  # d_lb = lb1 - ub2
                        elif 'B1' in ns_info.ns_type:
                            # update bounds
                            new_inp1_lb = inp1_lb  # lb1 = lb2 + d_lb
                            new_inp2_lb = inp2_lb  # lb2 = lb1 - d_ub
                            new_d_ub = d_ub  # d_ub = ub1 - lb2
                            new_inp2_ub = 0  # ub2 = 0
                            new_inp1_ub = min(inp1_ub, d_ub)  # ub1 = ub2 + d_ub
                            new_d_lb = max(d_lb, inp1_lb)  # d_lb = lb1 - ub2
                        elif 'B2' in ns_info.ns_type:
                            # update bounds
                            new_inp1_ub = inp1_ub  # ub1 = ub2 + d_ub
                            new_inp2_ub = inp2_ub  # ub2 = ub1 - d_lb
                            new_d_lb = d_lb  # d_lb = lb1 - ub2
                            new_inp2_lb = 0  # lb2 = 0
                            new_inp1_lb = max(inp1_lb, d_lb)  # lb1 = lb2 + d_lb
                            new_d_ub = min(d_ub, inp1_ub)  # d_ub = ub1 - lb2

                    self.inp1_lbs[layer_idx][ns_info.pos] = new_inp1_lb
                    self.inp1_ubs[layer_idx][ns_info.pos] = new_inp1_ub
                    self.inp2_lbs[layer_idx][ns_info.pos] = new_inp2_lb
                    self.inp2_ubs[layer_idx][ns_info.pos] = new_inp2_ub
                    self.d_lbs[layer_idx][ns_info.pos] = new_d_lb
                    self.d_ubs[layer_idx][ns_info.pos] = new_d_ub

            curr_inp1_lb, curr_inp1_ub, curr_inp2_lb, curr_inp2_ub, curr_d_lb, curr_d_ub = self.back_substitution_IAR(layer_idx=layer_idx)
            if curr_inp1_lb is None:
                return None

            self.inp1_lbs.append(curr_inp1_lb)
            self.inp1_ubs.append(curr_inp1_ub)
            self.inp2_lbs.append(curr_inp2_lb)
            self.inp2_ubs.append(curr_inp2_ub)
            self.d_lbs.append(curr_d_lb)
            self.d_ubs.append(curr_d_ub)

        return self

    # $ Run IAR backsubstitution for all layers
    def run_full_back_substitution_IAR(self):
        """
        Runs the full back substitution for Individual And Relational bounds.
        """

        self.inp1_lbs = [self.inp1_input_lb]
        self.inp1_ubs = [self.inp1_input_ub]
        self.inp2_lbs = [self.inp2_input_lb]
        self.inp2_ubs = [self.inp2_input_ub]
        if self.inp1_relu_input_info is None:
            self.inp1_relu_input_info = [None]
            self.inp2_relu_input_info = [None]

        if self.diff is not None:
            self.d_lbs = [self.diff]
            self.d_ubs = [self.diff]
        if self.delta_eps is not None:
            self.d_lbs = [torch.full_like(self.inp1_input_lb, -self.delta_eps, device=self.device)]
            self.d_ubs = [torch.full_like(self.inp1_input_ub, self.delta_eps, device=self.device)]

        # Check the validity of inputs.
        if self.net is None:
            raise ValueError("Passed network can not be none")

        for layer_idx, layer in enumerate(self.net):
            curr_inp1_lb, curr_inp1_ub, curr_inp2_lb, curr_inp2_ub, curr_d_lb, curr_d_ub = self.back_substitution_IAR(layer_idx=layer_idx)
            if curr_inp1_lb is None:  # infeasible bounds detected
                return None, None, None, None, None, None

            self.inp1_lbs.append(curr_inp1_lb)
            self.inp1_ubs.append(curr_inp1_ub)
            self.inp2_lbs.append(curr_inp2_lb)
            self.inp2_ubs.append(curr_inp2_ub)
            self.d_lbs.append(curr_d_lb)
            self.d_ubs.append(curr_d_ub)

            if len(self.inp1_relu_input_info) + 1 == len(self.inp1_lbs):
                self.inp1_relu_input_info.append(None)
                self.inp2_relu_input_info.append(None)
            else:
                raise ValueError("ReLU input information is not consistent.")
        return self.inp1_lbs, self.inp1_ubs, self.inp2_lbs, self.inp2_ubs, self.d_lbs, self.d_ubs

    def run(self):
        with torch.no_grad():
            for ind, layer in enumerate(self.net):
                if layer.type in [LayerType.Linear, LayerType.Conv2D]:
                    self.linear_conv_layer_indices.append(ind)

            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n\n## IAR start\n")

            return self.run_full_back_substitution_IAR()
