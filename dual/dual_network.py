import gc
import torch
from common.network import LayerType
from dual.dual_layers import DualLinear, DualConv2D, DualRelu
from dual.dual_analysis import DualAnalysis


def get_relational_order(net, C, DS_mode, layer_idx, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
    dualnet_1 = DualNetwork(C)
    dualnet_1.build_dual_network_relational(net, C, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)
    dualnet_2 = DualNetwork(-C)
    dualnet_2.build_dual_network_relational(net, -C, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)
    if DS_mode is not None:
        # get ds order
        dual_analysis = DualAnalysis()
        ds_order = dual_analysis.estimate_relational_impact(dualnet_1, dualnet_2, DS_mode, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)

        # free up memory
        del dualnet_1
        del dualnet_2
        del dual_analysis
        gc.collect()
        torch.cuda.empty_cache()

        return ds_order  # ds_order: [[ds_type, lubs_layer_idx, pos], ...]


def get_relational_order_ns(net, C, NS_mode, layer_idx, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
    dualnet_1 = DualNetwork(C)
    dualnet_1.build_dual_network_relational(net, C, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)
    dualnet_2 = DualNetwork(-C)
    dualnet_2.build_dual_network_relational(net, -C, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)
    if NS_mode is not None:
        # get ns order
        dual_analysis = DualAnalysis()
        ns_order = dual_analysis.estimate_relational_impact_ns(dualnet_1, dualnet_2, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs)

        # free up memory
        del dualnet_1
        del dualnet_2
        del dual_analysis
        gc.collect()
        torch.cuda.empty_cache()

        return ns_order  # ns_order: [[ns_type, lubs_layer_idx, pos], ...]


class DualNetwork():  # for relational
    def __init__(self, C):
        self.dual_net = []
        self.As_inp1 = []  # for relational
        self.As_inp2 = []  # for relational
        self.As_delta = []  # for relational
        self.C = C
        self.dual_analysis = DualAnalysis()
        self.build_dual_network = self.build_dual_network_relational

    def build_dual_network_relational(self, net, C, input_shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        dual_net = []

        for layer_idx, layer in enumerate(net):
            if layer.type is LayerType.Linear:
                dual_layer = DualLinear(layer)
            elif layer.type is LayerType.Conv2D:
                # Assuming the network has alternate affine and activation layers.
                # input_shapes includes the shapes of the linear layers.
                linear_layer_index = layer_idx // 2
                dual_layer = DualConv2D(layer, in_shape=input_shapes[linear_layer_index], out_shape=input_shapes[linear_layer_index + 1])
            elif layer.type is LayerType.ReLU:
                lubs_idx = layer_idx
                dual_layer = DualRelu(inp1_lbs[lubs_idx], inp1_ubs[lubs_idx], inp2_lbs[lubs_idx], inp2_ubs[lubs_idx], d_lbs[lubs_idx], d_ubs[lubs_idx])
            else:
                raise ValueError(f"Unsupported layer type: {layer.type}")

            dual_net.append(dual_layer)
        self.dual_net = dual_net

        As_inp1 = [torch.zeros_like(inp1_lbs[-1])]
        As_inp2 = [torch.zeros_like(inp2_lbs[-1])]
        As_delta = [-C]  # depends on the objective (and minimization or maximization)

        for r_layer in reversed(dual_net):
            curr_As_inp1, curr_As_inp2, curr_As_delta = r_layer.T(As_inp1, As_inp2, As_delta)
            As_inp1.append(curr_As_inp1)
            As_inp2.append(curr_As_inp2)
            As_delta.append(curr_As_delta)

        self.As_inp1 = As_inp1
        self.As_inp2 = As_inp2
        self.As_delta = As_delta

        return
