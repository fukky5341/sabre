import copy
import torch
import random
from common import Status


class DS_handler:
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

    def get_ds_random_order(self, ds_mode, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        layer_idx indicates the index of the activation layer (ReLU layer) in the net
        layer_idx also indicates the index of the preactivation lubs in the lubs list
        net: [linear, relu, linear, relu, ...]
        lubs: [inp, linear, relu, linear, relu, ...]
        """
        ds_conditions = self.sort_into_single(inp1_lbs[layer_idx], inp1_ubs[layer_idx], inp2_lbs[layer_idx],
                                           inp2_ubs[layer_idx], d_lbs[layer_idx], d_ubs[layer_idx])
        if not ds_conditions['common'].any():
            return []

        temp_order = []
        if ds_mode == 'DS_random_Z':
            active_idx = torch.where(ds_conditions['DSZ'])[0]  # e.g., tensor([1, 2, 4, 5, ...])
            active_idx_list = active_idx.tolist()
            for idx in active_idx_list:
                temp_order.append(['DSZ', layer_idx, idx])

        if len(temp_order) == 0:
            return []
        random.shuffle(temp_order)
        ds_order = []
        seen_indices = set()
        for item in temp_order:  # item is [ds_type, layer_idx, pos]
            if item[2] not in seen_indices:
                candidate = [item[0], layer_idx, item[2]]
                ds_order.append(candidate)
                seen_indices.add(item[2])
        return ds_order
    
    def sort_into_single(self, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub):
        ds_conditions = {}
        # conditions for sorting
        ds_conditions['DSZ'] = ((d_lb < 0) & (0 < d_ub) & ((inp1_lb < -1e-6) & (1e-6 < inp1_ub))) | \
                ((d_lb < 0) & (0 < d_ub) & ((inp2_lb < -1e-6) & (1e-6 < inp2_ub)))
        ds_conditions['common'] = ds_conditions['DSZ']

        return ds_conditions

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
            elif ds.status == Status.ADV_EXAMPLE:
                return Status.ADV_EXAMPLE, ds.inp1_label, ds.inp2_label
            elif ds.status == Status.UNREACHABLE:
                continue
            else:
                return Status.UNKNOWN, None, None

        return status, inp1_label, inp2_label
