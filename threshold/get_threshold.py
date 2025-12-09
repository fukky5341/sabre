def get_thresholds(net_name, d_eps, i_eps, net_idx1=None, net_idx2=None):
    if 'mnist-net_256x4' in net_name:
        file_path = f"threshold/mnist4/d{d_eps}_e{i_eps}.txt"
    elif 'mnist_conv' in net_name:
        file_path = f"threshold/mnist-conv/d{d_eps}_e{i_eps}.txt"
    elif 'cifar10' in net_name:
        file_path = f"threshold/cifar10/d{d_eps}_e{i_eps}.txt"
    elif 'acasxu' in net_name:
        file_path = f"threshold/acasxu/net_{net_idx1}_{net_idx2}_d_{d_eps}.txt"

    return read_thresholds(file_path)


def read_thresholds(file_path):
    """
    e.g.,
    0.002742
    0.001629
    0.006227
    0.010318
    0.00135755
    """

    thresholds = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                thresholds.append(float(s))
            except ValueError:
                # skip lines that don't contain a single float
                continue
    return thresholds
