import pandas as pd
import torch
import torchvision
from torchvision import transforms
from common.dataset import Dataset
from specs.property import Property
from specs.input_spec import InputSpecType
from specs.out_spec import OutSpecType, Constraint
from specs.properties.acasxu import get_acas_spec_glb_rbst
import util

'''
Specification holds upper bound and lower bound on ranges for each dimension.
In future, it can be extended to handle other specs such as those for rotation 
or even the relu stability could be part of specification.
'''


def prepare_data(dataset, train=False, batch_size=100, normalize=False, shuffle=False, generator=None):
    transform_list = [torchvision.transforms.ToTensor()]

    if normalize:
        mean, std = get_mean_std(dataset)
        transform_list.append(torchvision.transforms.Normalize(mean=mean, std=std))

    tr = torchvision.transforms.Compose(transform_list)

    """
    test_set → (input_image_tensor, label)
    """
    if dataset == Dataset.CIFAR10 or dataset == Dataset.OVAL_CIFAR:
        test_set = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=tr)
    elif dataset == Dataset.MNIST:
        test_set = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=tr)
    else:
        raise ValueError("Unsupported Dataset")

    """
    testloader: batch together test_set
    """
    if generator is not None:
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, generator=generator)
    else:
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    return testloader


def process_input_for_target_label(inputs, labels, target_label, target_count=0):
    '''
    pick inputs with a label is a target label
    '''
    new_inputs = []
    new_labels = []
    count = 0
    if target_label is None:
        return inputs, labels
    for i in range(len(inputs)):
        if labels[i].item() == target_label and count < target_count:  # $ changed from "is" to "=="
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
            count += 1
    new_inputs = torch.stack(new_inputs)
    new_labels = torch.stack(new_labels)
    return new_inputs, new_labels


# def remove_unclassified_images(inputs, labels, dataset, net_name):
#     if net_name == '':
#         return inputs, labels

#     model = get_net(net_name, dataset)
#     try:
#         with torch.no_grad():
#             only_net_name = net_name.split('/')[-1]
#             converted_model = convert_model(model, remove_last_layer=False, all_linear=is_linear(net_name=only_net_name))
#             mean, std = get_mean_std(dataset=dataset, net_name=net_name)
#             norm_transform = norm(mean, std)
#             inputs_normalised = norm_transform(inputs)
#             outputs = converted_model(inputs_normalised)
#             output_labels = torch.max(outputs, axis=1)[1]
#             # print(f'matching tensor {output_labels == labels}')
#             inputs = inputs[output_labels == labels]
#             labels = labels[output_labels == labels]
#             return inputs, labels
#     except:
#         return inputs, labels


def get_specs(dataset, spec_type=InputSpecType.LINF, eps=0.01, count=None, shuffle=False, generator=None):
    # if debug_mode == True:
    #     return generate_debug_specs(count=count, eps=eps)
    if dataset == Dataset.MNIST or dataset == Dataset.CIFAR10:
        if spec_type == InputSpecType.LINF:
            if count is None:
                count = 100
            testloader = prepare_data(dataset, batch_size=count, shuffle=shuffle, generator=generator)  # * format and divide into batches
            inputs, labels = next(iter(testloader))
            props = get_linf_spec(inputs, labels, eps, dataset)  # * L-inf norm spec
        elif spec_type == InputSpecType.PATCH:
            if count is None:
                count = 10
            testloader = prepare_data(dataset, batch_size=count, shuffle=shuffle, generator=generator)
            inputs, labels = next(iter(testloader))
            props = get_patch_specs(inputs, labels, eps, dataset, p_width=2, p_length=2)
            width = inputs.shape[2] - 2 + 1
            length = inputs.shape[3] - 2 + 1
            pos_patch_count = width * length
            specs_per_patch = pos_patch_count
            # labels = labels.unsqueeze(1).repeat(1, pos_patch_count).flatten()
        return props, inputs

    elif dataset == Dataset.ACAS:
        if count is None or count > 10:
            count = 10
        props = get_acas_props(count)
        return props, None


# Get the specification for local linf robusteness.
# Untargeted uap are exactly same for local linf specs.
def get_linf_spec(inputs, labels, eps, dataset, net_name=''):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = torch.clip(image - eps, min=0., max=1.)
        iub = torch.clip(image + eps, min=0., max=1.)

        mean, std = get_mean_std(dataset, net_name=net_name)
        ilb = (ilb - mean) / std
        iub = (iub - mean) / std

        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image))

    return properties


def get_oval_cifar_props(count):
    pdprops = 'base_easy.pkl'  # pdprops = 'base_med.pkl' or pdprops = 'base_hard.pkl'
    path = 'data/cifar_exp/'
    gt_results = pd.read_pickle(path + pdprops)
    # batch ids were used for parallel processing in the original paper.
    batch_ids = gt_results.index[0:count]
    props = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                                              transform=transforms.Compose([transforms.ToTensor(), normalize]))
    for new_idx, idx in enumerate(batch_ids):
        imag_idx = gt_results.loc[idx]['Idx']
        adv_label = gt_results.loc[idx]['prop']
        eps_temp = gt_results.loc[idx]['Eps']

        ilb, iub, true_label = util.ger_property_from_id(imag_idx, eps_temp, cifar_test)
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=true_label, adv_label=adv_label)
        props.append(Property(ilb, iub, InputSpecType.LINF, out_constr, Dataset.CIFAR10))
    return props, None


def get_acas_props(count):
    props = []
    if count is None:
        count = 10
    for i in range(1, count + 1):
        props.append(get_acas_spec_glb_rbst(i))
    return props


def get_binary_uap_spec(inputs, labels, eps, dataset, net_name=''):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = torch.clip(image - eps, min=0., max=1.)
        iub = torch.clip(image + eps, min=0., max=1.)

        mean, std = get_mean_std(dataset, net_name=net_name)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)

        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i], is_binary=True)
        properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image))

    return properties


def get_targeted_UAP_spec(inputs, labels, eps, dataset, net_name=''):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = torch.clip(image - eps, min=0., max=1.)
        iub = torch.clip(image + eps, min=0., max=1.)

        mean, std = get_mean_std(dataset, net_name=net_name)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)

        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        properties.append(Property(ilb, iub, InputSpecType.UAP, out_constr, dataset, input=image, targeted=True))

    return properties


def get_patch_specs(inputs, labels, eps, dataset, p_width=2, p_length=2):
    width = inputs.shape[2] - p_width + 1
    length = inputs.shape[3] - p_length + 1
    pos_patch_count = width * length
    final_bound_count = pos_patch_count

    patch_idx = torch.arange(pos_patch_count, dtype=torch.long)[None, :]

    x_cord = torch.zeros((1, pos_patch_count), dtype=torch.long, requires_grad=False)
    y_cord = torch.zeros((1, pos_patch_count), dtype=torch.long, requires_grad=False)
    idx = 0
    for w in range(width):
        for l in range(length):
            x_cord[0, idx] = w
            y_cord[0, idx] = l
            idx = idx + 1

    # expand the list to include coordinates from the complete patch
    patch_idx = [patch_idx.flatten()]
    x_cord = [x_cord.flatten()]
    y_cord = [y_cord.flatten()]
    for w in range(p_width):
        for l in range(p_length):
            patch_idx.append(patch_idx[0])
            x_cord.append(x_cord[0] + w)
            y_cord.append(y_cord[0] + l)

    patch_idx = torch.cat(patch_idx, dim=0)
    x_cord = torch.cat(x_cord, dim=0)
    y_cord = torch.cat(y_cord, dim=0)

    # create masks for each data point
    mask = torch.zeros([1, pos_patch_count, inputs.shape[2], inputs.shape[3]],
                       dtype=torch.uint8)
    mask[:, patch_idx, x_cord, y_cord] = 1
    mask = mask[:, :, None, :, :]
    mask = mask.cpu()

    iubs = torch.clip(inputs + 1, min=0., max=1.)
    ilbs = torch.clip(inputs - 1, min=0., max=1.)

    iubs = torch.where(mask, iubs[:, None, :, :, :], inputs[:, None, :, :, :])
    ilbs = torch.where(mask, ilbs[:, None, :, :, :], inputs[:, None, :, :, :])

    mean, stds = get_mean_std(dataset)

    iubs = (iubs - mean) / stds
    ilbs = (ilbs - mean) / stds

    # (data, patches, spec)
    iubs = iubs.view(iubs.shape[0], iubs.shape[1], -1)
    ilbs = ilbs.view(ilbs.shape[0], ilbs.shape[1], -1)

    props = []

    for i in range(ilbs.shape[0]):
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        props.append(Property(ilbs[i], iubs[i], InputSpecType.PATCH, out_constr, dataset, input=(inputs[i]-mean)/stds))
    return props


# todo check the cases when dataset == Dataset.MNIST
# def get_mean_std(dataset, net_name=''):
#     if dataset == Dataset.MNIST:
#         if 'crown' in net_name or is_linear_model(net_name) :
#             means = [0.0]
#             stds = [1.0]
#         else:
#             means = [0.1307]
#             stds = [0.3081]
#     elif dataset == Dataset.CIFAR10:
#         # For the model that is loaded from cert def this normalization was
#         # used
#         stds = [0.2023, 0.1994, 0.2010]
#         means = [0.4914, 0.4822, 0.4465]
#         # means = [0.0, 0.0, 0.0]
#         # stds = [1, 1, 1]
#     elif dataset == Dataset.ACAS:
#         means = [19791.091, 0.0, 0.0, 650.0, 600.0]
#         stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
#     elif dataset == Dataset.HOUSING:
#         means = 0
#         stds = 1
#     else:
#         raise ValueError("Unsupported Dataset!")
#     return torch.tensor(means).reshape(-1, 1, 1), torch.tensor(stds).reshape(-1, 1, 1)

def get_mean_std(dataset, net_name=''):
    if dataset == Dataset.MNIST:
        means = [0]
        stds = [1]
    elif dataset == Dataset.CIFAR10 or dataset == Dataset.OVAL_CIFAR:
        # For the model that is loaded from cert def this normalization was
        # used
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2023, 0.1994, 0.2010]
        # means = [0.5, 0.5, 0.5]
        # stds = [1, 1, 1]
    elif dataset == Dataset.ACAS:
        means = [19791.091, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    else:
        raise ValueError("Unsupported Dataset!")
    return torch.tensor(means).reshape(-1, 1, 1), torch.tensor(stds).reshape(-1, 1, 1)
