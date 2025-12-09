import copy
import torch
import random
from common.network import LayerType
from common import Status
import torch.nn as nn


class NS_handler:
    def __init__(self):
        self.start_layer_idx = 0

    def duplicate_DS(self):
        ds = copy.deepcopy(self)
        ds.IARb = self.duplicate_IARb(self.IARb)
        return ds

    def duplicate_IARb(self, IARb):
        new_IARb = copy.deepcopy(IARb)
        new_IARb.inp1_relu_input_info = copy.deepcopy(IARb.inp1_relu_input_info)
        new_IARb.inp2_relu_input_info = copy.deepcopy(IARb.inp2_relu_input_info)
        return new_IARb

    def copy_grb_model(self, ds):
        if self.IARb.global_robustness_lp.grb_model is not None:
            ds.IARb.global_robustness_lp.grb_model = self.IARb.global_robustness_lp.grb_model.copy()

    def get_ns_random_order(self, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        layer_idx indicates the index of the activation layer (ReLU layer) in the net
        layer_idx also indicates the index of the preactivation lubs in the lubs list
        net: [linear, relu, linear, relu, ...]
        lubs: [inp, linear, relu, linear, relu, ...]
        """
        inp1_unstable = (inp1_lbs[layer_idx] < -1e-6) & (1e-6 < inp1_ubs[layer_idx])
        inp2_unstable = (inp2_lbs[layer_idx] < -1e-6) & (1e-6 < inp2_ubs[layer_idx])

        if not (inp1_unstable | inp2_unstable).any():
            return []
        ds_order = []
        for ns_inp, inp_i_unstable in zip(['A', 'B'], [inp1_unstable, inp2_unstable]):  # ns_inp: A or B
            active_idx = torch.where(inp_i_unstable)[0]
            for idx in active_idx:
                # pos = tuple(idx.tolist())
                pos = idx.item()
                ds_order.append([ns_inp, layer_idx, pos])
        random.shuffle(ds_order)
        return ds_order

    def get_NS_status(self, NS_position=None, NS_info=None, IARb=None):
        if NS_info is None:
            ns_inp, layer_idx, pos = NS_position[0], NS_position[1], NS_position[2]
        else:
            ns_inp = NS_info.ns_inp
            layer_idx = NS_info.layer_idx
            pos = NS_info.pos

        inp1_lb = IARb.inp1_lbs[layer_idx][pos]
        inp1_ub = IARb.inp1_ubs[layer_idx][pos]
        inp2_lb = IARb.inp2_lbs[layer_idx][pos]
        inp2_ub = IARb.inp2_ubs[layer_idx][pos]
        d_lb = IARb.d_lbs[layer_idx][pos]
        d_ub = IARb.d_ubs[layer_idx][pos]
        with open(f"{self.log_file}log.md", 'a') as f:
            f.write(f"\nNS status\n")
            f.write(f"position: ( {layer_idx}, {pos} )\n")
            f.write(f"layer type: {IARb.net[layer_idx -1].type}\n\n")
            f.write(f"inp1 (lb, ub): ({inp1_lb}, {inp1_ub})\n")
            f.write(f"inp2 (lb, ub): ({inp2_lb}, {inp2_ub})\n")
            f.write(f"d (lb, ub): ({d_lb}, {d_ub})\n")

        return inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub

    def collect_final_status(self, DS_list):
        status = Status.UNKNOWN
        inp1_label = None
        inp2_label = None
        for ds in DS_list:
            if ds.status == Status.VERIFIED:
                status = Status.VERIFIED
                if inp1_label is None:
                    inp1_label = ds.inp1_label
                else:
                    if inp1_label != ds.inp1_label:
                        raise ValueError(f"Multiple inp1 labels found: {inp1_label} and {ds.inp1_label}.")
                if inp2_label is None:
                    inp2_label = ds.inp2_label
                else:
                    if inp2_label != ds.inp2_label:
                        raise ValueError(f"Multiple inp2 labels found: {inp2_label} and {ds.inp2_label}.")
            elif ds.status == Status.UNREACHABLE:
                continue
            else:
                return Status.UNKNOWN, None, None

        return status, inp1_label, inp2_label
