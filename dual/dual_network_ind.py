import gc
import torch
from common.network import LayerType
from dual.dual_layers_ind import DualLinear_Ind, DualConv2D_Ind, DualRelu_Ind
from dual.dual_analysis_ind import DualAnalysis_Ind


def get_relational_order_ns_ind(net, C, NS_mode, layer_idx, input_shapes, inp1_lbs, inp1_ubs):
    dualnet_1 = DualNetwork_Ind(C)
    dualnet_1.build_dual_network_individual(net, C, input_shapes, inp1_lbs, inp1_ubs)
    dualnet_2 = DualNetwork_Ind(-C)
    dualnet_2.build_dual_network_individual(net, -C, input_shapes, inp1_lbs, inp1_ubs)
    if NS_mode is not None:
        # get ns order
        dual_analysis = DualAnalysis_Ind()
        ns_order = dual_analysis.estimate_individual_impact(dualnet_1, dualnet_2, layer_idx, inp1_lbs, inp1_ubs)

        # free up memory
        del dualnet_1
        del dualnet_2
        del dual_analysis
        gc.collect()
        torch.cuda.empty_cache()

        return ns_order  # ns_order: [[ns_type, lubs_layer_idx, pos], ...]


class DualNetwork_Ind():  # for individual
    def __init__(self, C):
        self.dual_net = []
        self.As = []  # for individual
        self.C = C
        self.dual_analysis = DualAnalysis_Ind()
        self.build_dual_network = self.build_dual_network_individual

    def build_dual_network_individual(self, net, C, input_shapes, lbs, ubs):
        dual_net = []

        for layer_idx, layer in enumerate(net):
            if layer.type is LayerType.Linear:
                dual_layer = DualLinear_Ind(layer)
            elif layer.type is LayerType.Conv2D:
                # Assuming the network has alternate affine and activation layers.
                # input_shapes includes the shapes of the linear layers.
                linear_layer_index = layer_idx // 2
                dual_layer = DualConv2D_Ind(layer, in_shape=input_shapes[linear_layer_index], out_shape=input_shapes[linear_layer_index + 1])
            elif layer.type is LayerType.ReLU:
                lubs_idx = layer_idx
                dual_layer = DualRelu_Ind(lbs[lubs_idx], ubs[lubs_idx])
            else:
                raise ValueError(f"Unsupported layer type: {layer.type}")

            dual_net.append(dual_layer)
        self.dual_net = dual_net

        As = [-C]  # depends on the objective (and minimization or maximization)

        for r_layer in reversed(dual_net):
            curr_As = r_layer.T(As)
            As.append(curr_As)

        self.As = As

        return
