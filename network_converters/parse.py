import torch
import common as common
# import util as util

from torch.nn import ReLU, Linear, Conv2d, Sigmoid, Tanh

from onnx import numpy_helper
from common.network import Layer, LayerType, Network


def get_transformer(transformer, net, prop, relu_mask=None):

    # For all abstraction based domains (all other domains than lp)
    transformer = forward_layers(net, relu_mask, transformer)

    return transformer


def forward_layers(net, relu_mask, transformers):
    for layer in net:
        if layer.type == LayerType.ReLU:
            transformers.handle_relu(layer)
        elif layer.type == LayerType.Linear:
            if layer == net[-1]:
                transformers.handle_linear(layer, last_layer=True)
            else:
                transformers.handle_linear(layer)
        elif layer.type == LayerType.Conv2D:
            transformers.handle_conv2d(layer)
        elif layer.type == LayerType.Normalization:
            transformers.handle_normalization(layer)
        elif layer.type == LayerType.Sigmoid:
            transformers.handle_sigmoid(layer)
        elif layer.type == LayerType.TanH:
            transformers.handle_tanh(layer)
    return transformers


def resolve_initializer_tensor(name, initializer_map, node_output_map):

    if name in initializer_map:
        return initializer_map[name]

    if name in node_output_map:
        node = node_output_map[name]
        if node.op_type == "Identity":
            return resolve_initializer_tensor(node.input[0], initializer_map, node_output_map)

    raise KeyError(f"Could not resolve tensor: {name}")


def parse_onnx_layers(net):
    def attr_map(node):
        return {a.name: a for a in node.attribute}

    input_shape = tuple([dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim])
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape)

    # Create a dict for initializers (weights/biases) and Constant nodes
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals))
                              for init_vals in net.graph.initializer}

    nodes = list(net.graph.node)

    # Constant nodes (sometimes weights or biases are given as Constant, not initializer)
    for node in nodes:
        if node.op_type == 'Constant':
            for a in node.attribute:
                if a.name == 'value':
                    model_name_to_val_dict[node.output[0]] = torch.tensor(numpy_helper.to_array(a.t))

    # For quick consumer lookup: output name → list of nodes
    from collections import defaultdict
    consumers = defaultdict(list)
    for idx, node in enumerate(nodes):
        for inp in node.input:
            consumers[inp].append((idx, node))

    final_activation = False

    for i, node in enumerate(nodes):
        op = node.op_type
        nd_inps = list(node.input)
        final_activation = False

        if op == 'Flatten':
            # skip
            continue

        elif op == 'MatMul':
            # Automatically determine which input is the initializer (weight) and which is the activation
            weight_name = next((x for x in nd_inps if x in model_name_to_val_dict), None)
            act_name = next((x for x in nd_inps if x not in model_name_to_val_dict), None)

            if weight_name is None or act_name is None:
                raise ValueError(f"Could not determine weight/activation from MatMul inputs: inputs={nd_inps}")

            W = model_name_to_val_dict[weight_name].detach().clone()

            # Bias is obtained if a subsequent Add node takes the MatMul output and an initializer as input
            bias = None
            out_name = node.output[0]
            for (j, n2) in consumers.get(out_name, []):
                if n2.op_type == 'Add':
                    other = n2.input[0] if n2.input[1] == out_name else n2.input[1]
                    if other in model_name_to_val_dict:
                        bias = model_name_to_val_dict[other].detach().clone()
                        break
            # If not found, use zero bias
            # If unsure about the shape, it's OK to let Layer handle None→zero
            if bias is None:
                if W.ndim == 2:
                    # If X @ W, out_dim is W.shape[1]; if W @ X, out_dim is W.shape[0]
                    out_dim = W.shape[1] if act_name == nd_inps[0] else W.shape[0]
                    bias = torch.zeros(out_dim, dtype=W.dtype)
                else:
                    bias = torch.zeros(W.shape[0], dtype=W.dtype)

            # Align weight orientation to Layer's expected shape (out_features, in_features)
            # Generally, MatMul is X @ W (X: [N,K], W: [K,M]) → transpose W to (M,K)
            if act_name == nd_inps[0]:  # If X is the first input (standard)
                if W.ndim == 2:
                    W = W.t()
            else:
                # W @ X form (rare) → already considered as (out, in)
                pass

            layer = Layer(weight=W, bias=bias, type=LayerType.Linear)
            layers.append(layer)

        # elif op == 'Gemm':
        #     # Gemm: Y = alpha * A * B + beta * C (in most cases, transB=1 and it is equivalent to Linear)
        #     A, B = nd_inps[0], nd_inps[1]
        #     C = nd_inps[2] if len(nd_inps) > 2 else None
        #     am = attr_map(node)
        #     transA = am.get('transA').i if 'transA' in am else 0
        #     transB = am.get('transB').i if 'transB' in am else 0
        #     W = model_name_to_val_dict[B].detach().clone()
        #     if transB:
        #         W = W.t()
        #     b = model_name_to_val_dict[C].detach().clone() if C in model_name_to_val_dict else torch.zeros(W.shape[0], dtype=W.dtype)
        #     # transA for A is usually 0 (handle here if needed)
        #     layer = Layer(weight=W, bias=b, type=LayerType.Linear)
        #     layers.append(layer)

        elif op == 'Gemm':
            # Gemm: Y = alpha * A * B + beta * C
            A, B = nd_inps[0], nd_inps[1]
            C = nd_inps[2] if len(nd_inps) > 2 else None
            am = attr_map(node)
            transA = am.get('transA').i if 'transA' in am else 0
            transB = am.get('transB').i if 'transB' in am else 0

            W = model_name_to_val_dict[B].detach().clone()

            # ★ここがポイント（Layer は (out,in) を期待）
            if transB == 1:
                # 計算では B^T が使われる → B は (out,in) のまま保持すればOK
                pass
            else:
                # 計算で B をそのまま使う → B は (in,out) なので (out,in) に転置して保持
                W = W.t()

            if C in model_name_to_val_dict:
                b = model_name_to_val_dict[C].detach().clone()
            else:
                b = torch.zeros(W.shape[0], dtype=W.dtype)

            layer = Layer(weight=W, bias=b, type=LayerType.Linear)
            layers.append(layer)

        elif op == 'Conv':
            # Handle cases where bias is omitted, and retrieve attributes by name (order is not specified)
            W = model_name_to_val_dict[nd_inps[1]].detach().clone()
            if len(nd_inps) > 2 and nd_inps[2] in model_name_to_val_dict:
                b = model_name_to_val_dict[nd_inps[2]].detach().clone()
            else:
                b = torch.zeros(W.shape[0], dtype=W.dtype)

            am = attr_map(node)
            k = tuple(am['kernel_shape'].ints) if 'kernel_shape' in am else tuple(W.shape[2:])
            s = tuple(am['strides'].ints) if 'strides' in am else (1, 1)
            d = tuple(am['dilations'].ints) if 'dilations' in am else (1, 1)
            pads = tuple(am['pads'].ints) if 'pads' in am else (0, 0, 0, 0)
            # pads is in the order [top, left, bottom, right] (for 2D). Convert if Layer expects (ph, pw).
            if len(pads) == 4:
                pad = (pads[0], pads[1])  # Adjust here if only passing one side
            else:
                pad = (pads[0], pads[1])

            layer = Layer(weight=W, bias=b, type=LayerType.Conv2D)
            layer.kernel_size = (k[0], k[1])
            layer.stride = (s[0], s[1])
            layer.padding = pad
            layer.dilation = (d[0], d[1])
            layers.append(layer)

        elif op == 'Relu':
            final_activation = True
            layers.append(Layer(type=LayerType.ReLU))

        elif op == 'Sigmoid':
            final_activation = True
            layers.append(Layer(type=LayerType.Sigmoid))

        elif op == 'Tanh':
            final_activation = True
            layers.append(Layer(type=LayerType.TanH))

        # （Reshape/Transpose/BatchNorm and so on）if you need...

    # Hack that removes the final activation, which can potentially cause issues
    # if final_activation is True:
    #     layers.pop()

    return layers


# def parse_onnx_layers(net):
#     input_shape = tuple([dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim])
#     # Create the new Network object
#     layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape)
#     num_layers = len(net.graph.node)
#     model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in
#                               net.graph.initializer}

#     final_activation = False
#     for cur_layer in range(num_layers):
#         final_activation = False
#         node = net.graph.node[cur_layer]
#         operation = node.op_type
#         nd_inps = node.input
#         if operation == 'MatMul':
#             # Assuming that the add node is followed by the MatMul node
#             add_node = net.graph.node[cur_layer + 1]
#             bias = model_name_to_val_dict[add_node.input[1]]

#             # Making some weird assumption that the weight is always 0th index
#             if 'cast_input' in nd_inps[0] or 'next_activations' in nd_inps[0]:  # weird change for adult tanh network
#                 nd_inps[0] = nd_inps[1]
#                 layer = Layer(weight=model_name_to_val_dict[nd_inps[0]].T, bias=bias.T, type=LayerType.Linear)
#             else:
#                 layer = Layer(weight=model_name_to_val_dict[nd_inps[0]], bias=bias, type=LayerType.Linear)
#             layers.append(layer)

#         elif operation == 'Conv':
#             layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
#                           type=LayerType.Conv2D)
#             layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
#             layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
#             layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
#             layer.dilation = (1, 1)
#             layers.append(layer)

#         elif operation == 'Gemm':
#             # Making some weird assumption that the weight is always 1th index
#             layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
#                           type=LayerType.Linear)
#             layers.append(layer)

#         elif operation == 'Relu':
#             final_activation = True
#             layers.append(Layer(type=LayerType.ReLU))

#         elif operation == 'Sigmoid':
#             final_activation = True
#             layers.append(Layer(type=LayerType.Sigmoid))
#         elif operation == 'Tanh':
#             final_activation = True
#             layers.append(Layer(type=LayerType.TanH))

#         # Handle operation Sigmoid and TanH.

#     # The final most layer is relu and no linear layer after that
#     # remove the linear layer (Hack find better solutions).
#     if final_activation is True:
#         layers.pop()
#     return layers


def parse_torch_layers(net):
    layers = Network(torch_net=net)

    for torch_layer in net:
        if isinstance(torch_layer, ReLU):
            layers.append(Layer(type=LayerType.ReLU))
        elif isinstance(torch_layer, Sigmoid):
            layers.append(Layer(type=LayerType.Sigmoid))
        elif isinstance(torch_layer, Tanh):
            layers.append(Layer(type=LayerType.TanH))
        elif isinstance(torch_layer, Linear):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Linear)
            layers.append(layer)
        elif isinstance(torch_layer, Conv2d):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias,
                          type=LayerType.Conv2D)
            layer.kernel_size = torch_layer.kernel_size
            layer.padding = torch_layer.padding
            layer.stride = torch_layer.stride
            layer.dilation = (1, 1)
            layers.append(layer)

    return layers
