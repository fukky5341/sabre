from common.network import LayerType, Layer
import torch
import onnx
import network_converters.parse as parse
import training.models as models
from common.nn_networks import FullyConnected, Conv
from network_converters.load_pytorch_net_helper import load_pth_model, load_pth_model_modified, load_pt_model_modified
from network_converters.binary_model_loader import load_model
from common.dataset import Dataset
from onnx2torch import convert


def compute_input_shapes(net, input_shape):
    """
    Return a list of "layer input/output shapes" flowing through the net.
    - For Conv2D layers: shape is (C,H,W)
    - For Linear layers: shape is an int (out_features)
    - For activations (ReLU/Sigmoid/Tanh...): this function ignores them (no append)
    """
    # helper functions
    def _as_tuple(sh):
        if isinstance(sh, int):
            return (sh,)
        return tuple(sh)

    def _numel(shape_tuple):
        n = 1
        for d in shape_tuple:
            n *= d
        return n

    shapes = []
    cur = _as_tuple(input_shape)  # e.g., (1,28,28) / (3,32,32) / (5,)

    shapes.append(cur)  # input shape

    for idx, layer in enumerate(net):
        if layer.type is LayerType.Linear:
            # flatten if first layer
            if idx == 0:
                if len(cur) == 3:
                    # (C,H,W) → flatten
                    cur = _numel(cur)  # int
                    shapes.pop()
                    shapes.append(cur)
                elif len(cur) == 1:
                    # already flattened
                    cur = cur[0]
                    shapes.pop()
                    shapes.append(cur)
                else:
                    # flatten
                    cur = _numel(cur)
                    shapes.pop()
                    shapes.append(cur)
            # keep Linear layer output shape
            shapes.append(layer.weight.shape[0])
            cur = layer.weight.shape[0]

        elif layer.type is LayerType.Conv2D:
            # assume Conv2D consists of (C,H,W)
            if isinstance(cur, int) or (isinstance(cur, tuple) and len(cur) != 3):
                raise ValueError(f"Conv2D expects (C,H,W), got {cur}")

            weight = layer.weight
            num_kernel = weight.shape[0]
            k_h, k_w = layer.kernel_size
            s_h, s_w = layer.stride
            p_h, p_w = layer.padding

            _, input_h, input_w = cur
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
            output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

            cur = (num_kernel, output_h, output_w)
            shapes.append(cur)

        else:
            # pass ReLU/Sigmoid/Tanh/Flatten/Reshape
            pass

    return shapes


# def compute_input_shapes(net, input_shape):
#     shapes = []
#     shapes.append(input_shape)
#     for idx, layer in enumerate(net):
#         if idx == 0 and layer.type is LayerType.Linear:
#             shapes.pop()
#             shapes.append(input_shape[0] * input_shape[1] * input_shape[2])
#         if layer.type is LayerType.Linear:
#             shapes.append(layer.weight.shape[0])
#         elif layer.type is LayerType.Conv2D:
#             weight = layer.weight
#             num_kernel = weight.shape[0]

#             k_h, k_w = layer.kernel_size
#             s_h, s_w = layer.stride
#             p_h, p_w = layer.padding

#             shape = shapes[-1]

#             input_h, input_w = shape[1:]

#             ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
#             output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
#             output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)
#             shapes.append((num_kernel, output_h, output_w))

#     return shapes


def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'pt'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format


def get_torch_test_net(net_name, path, device='cpu', input_size=28):
    if net_name == 'fc1':
        net = FullyConnected(device, input_size, [50, 10]).to(device)
    elif net_name == 'fc2':
        net = FullyConnected(device, input_size, [100, 50, 10]).to(device)
    elif net_name == 'fc3':
        net = FullyConnected(device, input_size, [100, 100, 10]).to(device)
    elif net_name == 'fc4':
        net = FullyConnected(device, input_size, [100, 100, 50, 10]).to(device)
    elif net_name == 'fc5':
        net = FullyConnected(device, input_size, [100, 100, 100, 10]).to(device)
    elif net_name == 'fc6':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 10]).to(device)
    elif net_name == 'fc7':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 100, 10]).to(device)
    elif net_name == 'conv1':
        net = Conv(device, input_size, [(16, 3, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv2':
        net = Conv(device, input_size, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv3':
        net = Conv(device, input_size, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(device)
    else:
        assert False

    net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return net.layers


def get_pth_model_formats():
    _pth_model_format = {}
    _pth_model_format["cifar_cnn_3layer_fixed_kernel_3_width_1_best"] = {"in_ch": 3, "in_dim": 32, "kernel_size": 3, "width": 8}
    _pth_model_format["mnist_cnn_3layer_fixed_kernel_3_width_1_best"] = {"in_ch": 1, "in_dim": 28, "kernel_size": 3, "width": 1}
    _pth_model_format["cifar_cnn_2layer_width_2_best"] = {"in_ch": 3, "in_dim": 32, "width": 2, "linear_size": 256}
    _pth_model_format["mnist_cnn_2layer_width_1_best"] = {"in_ch": 1, "in_dim": 28, "width": 1, "linear_size": 128}
    _pth_model_format["crown_cifar_cnn_3layer_fixed_kernel_3_width_1_best"] = {"in_ch": 3, "in_dim": 32, "kernel_size": 3, "width": 8}
    _pth_model_format["crown_mnist_cnn_3layer_fixed_kernel_3_width_1_best"] = {"in_ch": 1, "in_dim": 28, "kernel_size": 3, "width": 1}
    _pth_model_format["crown_cifar_cnn_2layer_width_2_best"] = {"in_ch": 3, "in_dim": 32, "width": 2, "linear_size": 256}
    _pth_model_format["crown_mnist_cnn_2layer_width_1_best"] = {"in_ch": 1, "in_dim": 28, "width": 1, "linear_size": 128}
    return _pth_model_format


def get_pth_model_parameter(net_name):
    model_param_dict = get_pth_model_formats()
    if net_name not in model_param_dict.keys():
        raise ValueError("Format corresponding to net name not preset")
    return model_param_dict[net_name]


def load_binary_net(net_name):
    model = load_model(net_name=net_name)
    return model


def get_net_name(net_file):
    if 'pth.tar' in net_file:
        net_name = net_file.split('/')[-1].split('_')[0]
    else:
        net_name = net_file.split('/')[-1].split('.')[-2]
    return net_name


def get_torch_net(net_file, dataset, device='cpu'):
    net_name = net_file.split('/')[-1].split('.')[-2]
    # net_name = get_net_name(net_file)

    if 'pth' in net_file:
        if 'modified' in net_file:
            model = load_pth_model_modified(net_file)
            return model
        param_dict = get_pth_model_parameter(net_name)
        model = load_pth_model(net_file, param_dict)
        return model

    if 'pt' in net_file:
        if 'binary' in net_file:
            model = load_binary_net(net_name=net_file)
            if 'relu' in net_file:
                model = [model.fc1, model.relu, model.fc2]
            if 'sigmoid' in net_file:
                model = [model.fc1, model.sigmoid, model.fc2]
            if 'tanh' in net_file:
                model = [model.fc1, model.tanh, model.fc2]
            return model
        model = load_pt_model_modified(net_file)
        return model

    if 'cpt' in net_file:
        return get_torch_test_net(net_name, net_file)

    if dataset == Dataset.MNIST:
        model = models.Models[net_name](in_ch=1, in_dim=28)
    elif dataset == Dataset.CIFAR10:
        model = models.Models[net_name](in_ch=3, in_dim=32)
    else:
        raise ValueError("Unsupported dataset")

    if 'kw' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    elif 'eran' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    else:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'])

    return model


def get_debug_network():
    network = []
    weight1 = torch.tensor([[1, -1], [-2, 1]], dtype=torch.float)
    weight2 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float)
    network.append(Layer(weight=weight1, bias=torch.zeros(2), type=LayerType.Linear))
    network.append(Layer(type=LayerType.ReLU))
    network.append(Layer(weight=weight2, bias=torch.zeros(2), type=LayerType.Linear))
    return network


def get_net(net_name, dataset, debug_mode=False):
    if debug_mode:
        return get_debug_network()
    net_format = get_net_format(net_name)
    if net_format == 'pt':
        # Load the model
        with torch.no_grad():
            net_torch = get_torch_net(net_name, dataset)
            net = parse.parse_torch_layers(net_torch)

    elif net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net_torch = convert(net_onnx)
        # print(type(net_torch), net_torch)
        net = parse.parse_onnx_layers(net_onnx)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name
    net.net_format = net_format
    net.net_torch = net_torch
    return net


def ger_property_from_id(imag_idx, eps_temp, cifar_test):
    x, y = cifar_test[imag_idx]
    x = x.unsqueeze(0)

    ilb = (x - eps_temp).flatten()
    iub = (x + eps_temp).flatten()

    return ilb, iub, torch.tensor(y)
