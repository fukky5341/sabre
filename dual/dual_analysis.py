import torch
import torch.nn as nn
import torch.nn.functional as F
from common.network import LayerType


class PreactivationBounds:
    def __init__(self, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub):
        self.inp1_lb = inp1_lb
        self.inp1_ub = inp1_ub
        self.inp2_lb = inp2_lb
        self.inp2_ub = inp2_ub
        self.d_lb = d_lb
        self.d_ub = d_ub
        self.inp1_positive = (inp1_lb >= 0) & (inp1_ub > 0)
        self.inp1_negative = (inp1_ub <= 0)
        self.inp1_unstable = ~(self.inp1_positive) & ~(self.inp1_negative)
        self.inp2_positive = (inp2_lb >= 0) & (inp2_ub > 0)
        self.inp2_negative = (inp2_ub <= 0)
        self.inp2_unstable = ~(self.inp2_positive) & ~(self.inp2_negative)
        self.delta_positive = (d_lb >= 0) & (d_ub > 0)
        self.delta_negative = (d_ub <= 0)
        self.delta_unstable = ~(self.delta_positive) & ~(self.delta_negative)


class DualAnalysis:
    def __init__(self):
        pass

    def estimate_relational_impact(self, dualnet_1, dualnet_2, DS_mode, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        (requirement)
        - dual variables for each layer
        - individual bounds for preactivation layers

        lubs: [input, linear(0), ReLU(1), linear(2), ReLU(3), ..., linear(k)] lubs_idx is (layer_idx + 1)
        network: linear(0) --> ReLU(1) --> Linear(2) --> ReLU(3) --> Linear(4) ... --> Linear(k)
        dual vars: [dualOutput, dualLinear(k), dualReLU(k-1), dualLinear(k-2), dualReLU(k-3), ... , dualLinear(0)]
        dual network: [dualLinear(0), dualReLU(1), dualLinear(2), dualReLU(3), ..., dualLinear(k), dualOutput]

        idx = layer_idx <-- indicates the relu layer in the net (not include input layer)
        # required bias is in the (idx-1)th linear layer of dual net
        # preactivation lubs index is idx
        # preactivation dual variable (B) index is idx + 1, which is in reversed dual vars
        # activated dual variable (A) index is idx, which is in reversed dual vars
        list = [1, 2, 3, 4] --> list[::-1] = [4, 3, 2, 1]

        score
        - bias term
            (A + A')^t bias <-- index of A, A' is idx
        - interception term
            lambda*B + lambda'*B' + lambda_d*B_d
        """

        idx = layer_idx
        split_score_list = []
        for dual_network in [dualnet_1, dualnet_2]:
            split_scores, ds_conditions, ds_types = self.get_ds_merged_scores(DS_mode, dual_network.dual_net[idx-1].bias, idx, inp1_lbs[idx], inp1_ubs[idx],
                                                                              inp2_lbs[idx], inp2_ubs[idx], d_lbs[idx], d_ubs[idx],
                                                                              dual_network.As_inp1[::-1][idx+1], dual_network.As_inp2[::-1][idx+1],
                                                                              dual_network.As_delta[::-1][idx+1], dual_network.As_inp1[::-1][idx],
                                                                              dual_network.As_inp2[::-1][idx])
            if split_scores is None:
                return []
            split_score_list.append(split_scores)

        # combine scores from dualnet_1 and dualnet_2
        combined_scores = {}
        for ds_type in ds_types:
            combined_scores[ds_type] = split_score_list[0][ds_type] + split_score_list[1][ds_type]

        # arrange in descending order
        ds_order = self.arrange_in_descending_order(ds_conditions, combined_scores, idx, ds_types)

        return ds_order  # ds_order: [[ds_type, lubs_layer_idx, pos], ...]

    def estimate_relational_impact_ns(self, dualnet_1, dualnet_2, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        (requirement)
        - dual variables for each layer
        - individual bounds for preactivation layers

        network: linear(0) --> ReLU(1) --> Linear(2) --> ReLU(3) --> Linear(4) ... --> Linear(k)
        dual network: [dualOutput, dualLinear(k), dualReLU(k-1), dualLinear(k-2), dualReLU(k-3), ... , dualLinear(0)]
            reversed order [dualLinear(0), dualReLU(1), dualLinear(2), dualReLU(3), ..., dualLinear(k), dualOutput]

        idx = layer_idx <-- indicates the relu layer in the net (not include input layer)
        # required bias is in the (idx-1)th linear layer
        # preactivation lubs index is idx
        # preactivation dual variable (B) index is idx + 1, which is in backward path
        # activated dual variable (A) index is idx, which is in backward path
        list = [1, 2, 3, 4] --> list[::-1] = [4, 3, 2, 1]

        score
        - bias term
            (A + A')^t bias <-- index of A, A' is idx
        - interception term
            lambda*B + lambda'*B' + lambda_d*B_d
        """

        idx = layer_idx
        split_scores_list = []
        for dual_network in [dualnet_1, dualnet_2]:
            split_scores, ns_conditions = self.get_ns_merged_scores(dual_network.dual_net[idx-1].bias, idx, inp1_lbs[idx], inp1_ubs[idx],
                                                                    inp2_lbs[idx], inp2_ubs[idx], d_lbs[idx], d_ubs[idx],
                                                                    dual_network.As_inp1[::-1][idx+1],
                                                                    dual_network.As_inp2[::-1][idx+1],
                                                                    dual_network.As_delta[::-1][idx+1],
                                                                    dual_network.As_inp1[::-1][idx],
                                                                    dual_network.As_inp2[::-1][idx])
            if split_scores is None:
                return []
            split_scores_list.append(split_scores)

        # combine scores from dualnet_1 and dualnet_2
        combined_scores = {}
        for ns_type in ['A', 'B']:
            combined_scores[ns_type] = split_scores_list[0][ns_type] + split_scores_list[1][ns_type]

        # arrange in descending order
        ns_order = self.arrange_in_descending_order_ns(ns_conditions, combined_scores, idx)

        return ns_order

    def get_ds_merged_scores(self, DS_mode, bias, preactivation_lubs_idx, inp1_lb, inp1_ub, inp2_lb, inp2_ub,
                             d_lb, d_ub, inp1_v, inp2_v, delta_v, next_inp1_v, next_inp2_v):
        """
        split method:
        - zero

        procedure:
        - sort
        - get bounds conditions
        - get bounds after splitting
        - get score (estimation) for each split
            score = old{(A + A')^t bias} - new{(A + A')^t bias} 
                    + new{(ulB/(u-l)) + (u'l'B'/(u'-l')) + (ulB/(u-l))_delta} 
                    - old{(ulB/(u-l)) + (u'l'B'/(u'-l')) + (ulB/(u-l))_delta}
        - arrange in descending order

        backward direction
        next_v A(i-1) <-- v B(i) <-- pre_v A(i)
        """
        PB = PreactivationBounds(inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
        if not PB.inp1_unstable.any() or not PB.inp2_unstable.any():
            return [], [], []
        if 'DS_dual_Z' == DS_mode:
            ds_conditions = self.sort_into_single(PB, 'DSZ')  # DSZ, common
            ds_types = ['DSZ']
        else:
            raise ValueError(f"Unsupported DS_mode: {DS_mode}")

        # old score
        if ds_conditions['common'].any():
            old_score = self.get_old_score(bias, ds_conditions['common'], PB, inp1_v, inp2_v, delta_v, next_inp1_v, next_inp2_v)
        else:
            return None, None, None
        # split score
        ds_scores = self.get_split_scores_ds(old_score, bias, PB, ds_conditions, inp1_v, inp2_v, delta_v)

        return ds_scores, ds_conditions, ds_types

    def get_ns_merged_scores(self, bias, preactivation_lubs_idx, inp1_lb, inp1_ub, inp2_lb, inp2_ub,
                             d_lb, d_ub, inp1_v, inp2_v, delta_v, next_inp1_v, next_inp2_v):
        """
        split method:
            - zero

        procedure:
        - get unstable relu
        - old score
        - new score
            score = old{(A + A')^t bias} - new{(A + A')^t bias} 
                    + new{(ulB/(u-l)) + (u'l'B'/(u'-l')) + (ulB/(u-l))_delta} 
                    - old{(ulB/(u-l)) + (u'l'B'/(u'-l')) + (ulB/(u-l))_delta}
        - arrange in descending order

        backward direction
        next_v A(i-1) <-- v B(i) <-- pre_v A(i)
        """
        PB = PreactivationBounds(inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
        if not PB.inp1_unstable.any() or not PB.inp2_unstable.any():
            return None, None
        # sort
        ns_conditions = self.sort_unstable(PB)
        # old score
        if ns_conditions['common'].any():
            old_score = self.get_old_score(bias, ns_conditions['common'], PB, inp1_v, inp2_v, delta_v, next_inp1_v, next_inp2_v)
        else:
            return []
        # split score
        ns_scores = self.get_split_scores_ns(old_score, bias, PB, ns_conditions, inp1_v, inp2_v, delta_v)

        return ns_scores, ns_conditions

    def sort_into_two(self, PB):  # PB = PreactivationBounds
        ds_conditions = {}

        ds_conditions['DSM'] = ((1e-6 < PB.d_ub - PB.d_lb) & ((PB.inp1_lb < -1e-6) & (1e-6 < PB.inp1_ub))) | \
            ((1e-6 < PB.d_ub - PB.d_lb) & ((PB.inp2_lb < -1e-6) & (1e-6 < PB.inp2_ub)))
        ds_conditions['DSZ'] = ((PB.d_lb < 0) & (0 < PB.d_ub) & ((PB.inp1_lb < -1e-6) & (1e-6 < PB.inp1_ub))) | \
            ((PB.d_lb < 0) & (0 < PB.d_ub) & ((PB.inp2_lb < -1e-6) & (1e-6 < PB.inp2_ub)))

        ds_conditions['common'] = ds_conditions['DSM'] | ds_conditions['DSZ']

        return ds_conditions

    def sort_into_single(self, PB, ds_type):  # PB = PreactivationBounds
        ds_conditions = {}
        # conditions for sorting
        if ds_type == 'DSZ':
            ds_conditions['DSZ'] = ((PB.d_lb < 0) & (0 < PB.d_ub) & ((PB.inp1_lb < -1e-6) & (1e-6 < PB.inp1_ub))) | \
                ((PB.d_lb < 0) & (0 < PB.d_ub) & ((PB.inp2_lb < -1e-6) & (1e-6 < PB.inp2_ub)))
        ds_conditions['common'] = ds_conditions[ds_type]

        return ds_conditions

    def sort_unstable(self, PB):
        ns_conditions = {}
        ns_conditions['A'] = PB.inp1_unstable
        ns_conditions['B'] = PB.inp2_unstable
        ns_conditions['common'] = ns_conditions['A'] | ns_conditions['B']

        return ns_conditions

    def get_updated_lambda(self, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub, delta_v):
        # note: delta_v is the preactivation dual variable in the backward path

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
            # $ (case 5) inp1 is unstable, inp2 is negative
            case_5 = inp1_unstable & inp2_negative
            l_inp1 = torch.where(case_5, temp_inp1, l_inp1)
            l_inp1_d = torch.where(case_5, temp_inp1, l_inp1_d)
            # $ (case 6) inp1 is negative, inp2 is unstable
            case_6 = inp1_negative & inp2_unstable
            l_inp2 = torch.where(case_6, temp_inp2, l_inp2)
            l_inp2_d = torch.where(case_6, -temp_inp2, l_inp2_d)
            # $ (case 7) inp1 is unstable, inp2 is positive
            case_7 = inp1_unstable & inp2_positive
            l_inp1 = torch.where(case_7, temp_inp1, l_inp1)
            l_inp1_d = torch.where(case_7, temp_inp1, l_inp1_d)
            l_inp2 = torch.where(case_7, torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2)
            l_inp2_d = torch.where(case_7, -torch.ones(inp2_lb.size(), device=inp2_lb.device), l_inp2_d)
            # $ (case 8) inp1 is positive, inp2 is unstable
            case_8 = inp1_positive & inp2_unstable
            l_inp1 = torch.where(case_8, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1)
            l_inp1_d = torch.where(case_8, torch.ones(inp1_lb.size(), device=inp1_lb.device), l_inp1_d)
            l_inp2 = torch.where(case_8, temp_inp2, l_inp2)
            l_inp2_d = torch.where(case_8, -temp_inp2, l_inp2_d)
            # $ (case 9) inp1 is unstable, inp2 is unstable, delta is positive
            case_9 = inp1_unstable & inp2_unstable & delta_positive
            l_inp1 = torch.where(case_9, temp_inp1, l_inp1)
            l_inp2 = torch.where(case_9, temp_inp2, l_inp2)
            case_9_pre_positive = case_9 & (delta_v > 0)
            l_d = torch.where(case_9_pre_positive, torch.ones(d_lb.size(), device=d_lb.device), l_d)
            # $ (case 10) inp1 is unstable, inp2 is unstable, delta is negative
            case_10 = inp1_unstable & inp2_unstable & delta_negative
            l_inp1 = torch.where(case_10, temp_inp1, l_inp1)
            l_inp2 = torch.where(case_10, temp_inp2, l_inp2)
            case_10_pre_negative = case_10 & (delta_v < 0)
            l_d = torch.where(case_10_pre_negative, torch.ones(d_lb.size(), device=d_lb.device), l_d)
            # $ (case 11) inp1 is unstable, inp2 is unstable, delta is unstable
            case_11 = inp1_unstable & inp2_unstable & delta_unstable
            if case_11.any():
                case_11_pre_negative = case_11 & (delta_v < 0)
                case_11_pre_positive = case_11 & (delta_v > 0)
                temp_d_lb = -d_lb / (d_ub - d_lb + 1e-15)
                temp_d_ub = d_ub / (d_ub - d_lb + 1e-15)
                l_d = torch.where(case_11_pre_negative, temp_d_lb, l_d)
                l_d = torch.where(case_11_pre_positive, temp_d_ub, l_d)

        return l_inp1, l_inp1_d, l_inp2, l_inp2_d, l_d

    def get_y_intercept_term_score(self, PB, inp1_v, inp2_v, delta_v):
        y_score = torch.zeros_like(inp1_v)

        inp1_v_positive = torch.clamp(inp1_v, min=0)  # e.g., inp1_v = [0.5, -0.5, 0.3, -0.2], then inp1_v_positive = [0.5, 0, 0.3, 0]
        inp2_v_positive = torch.clamp(inp2_v, min=0)
        delta_v_positive = torch.clamp(delta_v, min=0)
        delta_v_negative = torch.clamp(-delta_v, min=0)  # e.g., delta_v = [0.5, -0.5, 0.3, -0.2], then delta_v_negative = [0, 0.5, 0, 0.2]
        if PB.inp1_unstable.any():
            temp_c_inp1 = (PB.inp1_ub*PB.inp1_lb) / (PB.inp1_ub - PB.inp1_lb + 1e-15)
        if PB.inp2_unstable.any():
            temp_c_inp2 = (PB.inp2_ub*PB.inp2_lb) / (PB.inp2_ub - PB.inp2_lb + 1e-15)
        if PB.inp1_unstable.any() & PB.inp2_unstable.any():
            temp_c_delta = (PB.d_ub*PB.d_lb) / (PB.d_ub - PB.d_lb + 1e-15)
        # $ (case 5) inp1 is unstable, inp2 is negative
        case_5 = PB.inp1_unstable & PB.inp2_negative
        # $ (case 6) inp1 is negative, inp2 is unstable
        case_6 = PB.inp1_negative & PB.inp2_unstable
        # $ (case 7) inp1 is unstable, inp2 is positive
        case_7 = PB.inp1_unstable & PB.inp2_positive
        # $ (case 8) inp1 is positive, inp2 is unstable
        case_8 = PB.inp1_positive & PB.inp2_unstable

        case_5_7 = case_5 | case_7
        case_6_8 = case_6 | case_8
        if case_5_7.any():
            y_score = torch.where(case_5_7, temp_c_inp1*inp1_v_positive + temp_c_inp1*delta_v_positive, y_score)
        if case_6_8.any():
            y_score = torch.where(case_6_8, temp_c_inp2*inp2_v_positive + temp_c_inp2*delta_v_negative, y_score)

        # $ (case 9) inp1 is unstable, inp2 is unstable, delta is positive
        case_9 = PB.inp1_unstable & PB.inp2_unstable & PB.delta_positive
        # $ (case 10) inp1 is unstable, inp2 is unstable, delta is negative
        case_10 = PB.inp1_unstable & PB.inp2_unstable & PB.delta_negative
        # $ (case 11) inp1 is unstable, inp2 is unstable, delta is unstable
        case_11 = PB.inp1_unstable & PB.inp2_unstable & PB.delta_unstable

        case_9_10 = case_9 | case_10
        if case_9_10.any():
            y_score = torch.where(case_9_10, temp_c_inp1*inp1_v_positive + temp_c_inp2*inp2_v_positive, y_score)
        if case_11.any():
            y_score = torch.where(case_11, temp_c_inp1*inp1_v_positive + temp_c_inp2*inp2_v_positive
                                  + temp_c_delta*delta_v_positive + temp_c_delta*delta_v_negative, y_score)

        return y_score

    def get_old_score(self, bias, common_condition, PB, inp1_v, inp2_v, delta_v, next_inp1_v, next_inp2_v):
        """
        backward direction
        next_v A(i-1) <-- v B(i) <-- pre_v A(i)

        old score:
        - (A + A')^t bias <-- (i-1)th layer's A and bias
        - 
        """
        # bias term
        old_score = torch.where(common_condition, (next_inp1_v + next_inp2_v)*bias, torch.zeros_like(bias))

        # y-intercept term
        temp_old_score = self.get_y_intercept_term_score(PB, inp1_v, inp2_v, delta_v)

        temp_old_score = torch.where(common_condition, temp_old_score, torch.zeros_like(temp_old_score))
        old_score -= temp_old_score
        return old_score

    def calculate_score(self, split_condition, bias, inp1_lb, inp1_ub, inp2_lb, inp2_ub,
                        d_lb, d_ub, inp1_v, inp2_v, delta_v):

        l_inp1, l_inp1_d, l_inp2, l_inp2_d, l_d = self.get_updated_lambda(inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub, delta_v)
        updated_inp1_v = l_inp1 * inp1_v + l_inp1_d * delta_v
        updated_inp2_v = l_inp2 * inp2_v + l_inp2_d * delta_v
        updated_delta_v = l_d * delta_v
        score = torch.where(split_condition, (updated_inp1_v + updated_inp2_v)*bias, torch.zeros_like(bias))
        score = -1 * score

        temp_PB = PreactivationBounds(inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
        temp_score = self.get_y_intercept_term_score(temp_PB, inp1_v, inp2_v, delta_v)
        temp_score = torch.where(split_condition, temp_score, torch.zeros_like(temp_score))
        score += temp_score

        return score

    def get_split_scores_ds(self, old_score, bias, PB, ds_conditions, inp1_v, inp2_v, delta_v):
        """
        backward direction
        next_v A(i-1) <-- v B(i) <-- pre_v A(i)

        ds: [dl, du] --> (1)[dl, new_d], (2)[new_d, du]

        1. bounds after splitting
        2. update conditions
        3. get score for each child
        4. merge scores
        """
        split_scores = {}
        for ds_name, ds_condition in ds_conditions.items():
            new_d = None
            if ds_name == 'common':
                continue
            elif ds_name == 'DSZ':
                if ds_condition.any():
                    new_d = 0

            # get bounds after splitting
            if new_d is not None:
                # (1) [dl, new_d]
                d_lb_1 = torch.where(ds_condition, PB.d_lb, torch.zeros_like(PB.d_lb))  # d_lb = inp1_lb - inp2_ub
                d_ub_1 = torch.where(ds_condition, new_d, torch.zeros_like(PB.d_lb))  # d_ub = new_d
                inp1_lb_1 = torch.where(ds_condition, PB.inp1_lb, torch.zeros_like(PB.inp1_lb))  # inp1_lb = d_lb + inp2_lb
                inp2_ub_1 = torch.where(ds_condition, PB.inp2_ub, torch.zeros_like(PB.inp2_ub))  # inp2_ub = inp1_ub - d_lb
                inp1_ub_1 = torch.min(PB.inp1_ub, d_ub_1 + inp2_ub_1)  # inp1_ub = inp2_ub + new_d(d_ub)
                inp2_lb_1 = torch.max(PB.inp2_lb, inp1_lb_1 - d_ub_1)  # inp2_lb = inp1_lb - new_d(d_ub)
                score_1 = self.calculate_score(ds_condition, bias, inp1_lb_1, inp1_ub_1, inp2_lb_1, inp2_ub_1,
                                               d_lb_1, d_ub_1, inp1_v, inp2_v, delta_v)
                # (2) [new_d, du]
                d_lb = torch.where(ds_condition, new_d, torch.zeros_like(PB.d_lb))  # d_lb = new_d
                d_ub = torch.where(ds_condition, PB.d_ub, torch.zeros_like(PB.d_lb))  # d_ub = inp1_ub - inp2_lb
                inp1_ub = torch.where(ds_condition, PB.inp1_ub, torch.zeros_like(PB.inp1_ub))  # inp1_ub = d_ub + inp2_ub
                inp2_lb = torch.where(ds_condition, PB.inp2_lb, torch.zeros_like(PB.inp2_lb))  # inp2_lb = inp1_lb - d_ub
                inp1_lb = torch.max(PB.inp1_lb, inp2_lb + d_lb)  # inp1_lb = inp2_lb + new_d(d_lb)
                inp2_ub = torch.min(PB.inp2_ub, inp1_ub - d_lb)  # inp2_ub = inp1_ub - new_d(d_lb)
                score_2 = self.calculate_score(ds_condition, bias, inp1_lb, inp1_ub, inp2_lb, inp2_ub,
                                               d_lb, d_ub, inp1_v, inp2_v, delta_v)

                # merge scores
                split_score1 = score_1 + torch.where(ds_condition, old_score, torch.zeros_like(old_score))
                split_score2 = score_2 + torch.where(ds_condition, old_score, torch.zeros_like(old_score))
                # $ merge scores
                # ---- here, there are some ways to calculate the split score ----
                split_score = split_score1 + split_score2

                split_scores[ds_name] = split_score
            else:
                split_scores[ds_name] = torch.zeros_like(old_score),

        return split_scores

    def get_split_scores_ns(self, old_score, bias, PB, ns_conditions, inp1_v, inp2_v, delta_v):
        """
        backward direction
        next_v A(i-1) <-- v B(i) <-- pre_v A(i)

        ns: [il, iu] --> (1)[il, 0], (2)[0, iu]

        1. bounds after splitting
        2. update conditions
        3. get score for each child
        4. merge scores
        """
        split_scores = {}
        for ns_name, ns_condition in ns_conditions.items():
            if ns_name == 'common':
                continue
            elif ns_name == 'A':  # inp1
                if ns_condition.any():
                    # get bounds after splitting
                    # (1) inp1[il, 0]
                    inp1_lb_1 = PB.inp1_lb
                    inp2_lb_1 = PB.inp2_lb
                    d_lb_1 = PB.d_lb
                    inp1_ub_1 = torch.zeros_like(PB.inp1_ub)  # new_inp1_ub = 0
                    inp2_ub_1 = torch.min(PB.inp2_ub, -PB.d_lb)  # inp2_ub = inp1_ub - d_lb
                    d_ub_1 = torch.min(PB.d_ub, -PB.inp2_lb)  # d_ub = inp1_ub - inp2_lb
                    score_1 = self.calculate_score(ns_condition, bias, inp1_lb_1, inp1_ub_1, inp2_lb_1, inp2_ub_1,
                                                   d_lb_1, d_ub_1, inp1_v, inp2_v, delta_v)
                    # (2) inp1[0, iu]
                    inp1_ub_2 = PB.inp1_ub
                    inp2_ub_2 = PB.inp2_ub
                    d_ub_2 = PB.d_ub
                    inp1_lb_2 = torch.zeros_like(PB.inp1_lb)  # new_inp1_lb = 0
                    inp2_lb_2 = torch.max(PB.inp2_lb, -PB.d_ub)  # inp2_lb = inp1_lb - d_ub
                    d_lb_2 = torch.max(PB.d_lb, -PB.inp2_ub)  # d_lb = inp1_lb - inp2_ub
                    score_2 = self.calculate_score(ns_condition, bias, inp1_lb_2, inp1_ub_2, inp2_lb_2, inp2_ub_2,
                                                   d_lb_2, d_ub_2, inp1_v, inp2_v, delta_v)
                else:
                    continue
            elif ns_name == 'B':  # inp2
                if ns_condition.any():
                    # get bounds after splitting
                    # (1) inp2[il, 0]
                    inp1_lb_1 = PB.inp1_lb
                    inp2_lb_1 = PB.inp2_lb
                    d_ub_1 = PB.d_ub
                    inp2_ub_1 = torch.zeros_like(PB.inp2_ub)  # new_inp2_ub = 0
                    inp1_ub_1 = torch.min(PB.inp1_ub, PB.d_ub)  # inp1_ub = inp2_ub + d_ub
                    d_lb_1 = torch.max(PB.d_lb, PB.inp1_lb)  # d_lb = inp1_lb - inp2_ub
                    score_1 = self.calculate_score(ns_condition, bias, inp1_lb_1, inp1_ub_1, inp2_lb_1, inp2_ub_1,
                                                   d_lb_1, d_ub_1, inp1_v, inp2_v, delta_v)
                    # (2) inp2[0, iu]
                    inp1_ub_2 = PB.inp1_ub
                    inp2_ub_2 = PB.inp2_ub
                    d_lb_2 = PB.d_lb
                    inp2_lb_2 = torch.zeros_like(PB.inp2_lb)  # new_inp2_lb = 0
                    inp1_lb_2 = torch.max(PB.inp1_lb, PB.d_lb)  # inp1_lb = inp2_lb + d_lb
                    d_ub_2 = torch.max(PB.d_ub, PB.inp1_ub)  # d_ub = inp1_ub - inp2_lb
                    score_2 = self.calculate_score(ns_condition, bias, inp1_lb_2, inp1_ub_2, inp2_lb_2, inp2_ub_2,
                                                   d_lb_2, d_ub_2, inp1_v, inp2_v, delta_v)
                else:
                    continue
            else:
                raise ValueError("Unknown neuron-split name")
            # merge scores
            split_score1 = score_1 + torch.where(ns_condition, old_score, torch.zeros_like(old_score))
            split_score2 = score_2 + torch.where(ns_condition, old_score, torch.zeros_like(old_score))
            # $ merge scores
            # ---- here, there are some ways to calculate the split score ----
            split_score = split_score1 + split_score2

            split_scores[ns_name] = split_score
            # else:
            #     split_scores[ds_name] = torch.zeros_like(old_score)

        return split_scores

    def arrange_in_descending_order(self, ds_conditions, ds_scores, preactivation_lubs_idx, ds_types):
        """
        Arrange the scores in descending order.
        ds_scores consists of scores for each split condition.
        return: [['split_condition', score, position], ...]

        ds_types = ['DSA', 'DSB', 'DSC', 'DSD'] or ['DSA', 'DSB', 'DSC', 'DSD', 'DSM', 'DSZ'] or ['DSM', 'DSZ']
        """
        sorted_scores = []
        for ds in ds_types:
            active_idx = torch.where(ds_conditions[ds])[0]
            if active_idx.numel() > 0:
                for idx in active_idx:
                    sorted_scores.append([ds, idx.item(),  ds_scores[ds][idx].item()])  # [ds_type, pos, score]
        # Sort by score in descending order
        sorted_scores.sort(key=lambda x: x[2], reverse=True)
        sorted_order = [[item[0], preactivation_lubs_idx, item[1]] for item in sorted_scores]  # candidate is [ds_type, lubs_layer_idx, pos]

        return sorted_order

    def arrange_in_descending_order_ns(self, ns_conditions, ds_scores, preactivation_lubs_idx):
        """
        Arrange the scores in descending order.
        ds_scores consists of scores for each split condition.
        return: [['split_condition', score, position], ...]
        """
        ns_names = ['A', 'B']
        sorted_scores = []
        for ns in ns_names:
            active_idx = torch.where(ns_conditions[ns])[0]
            if active_idx.numel() > 0:
                for idx in active_idx:
                    sorted_scores.append([ns, idx.item(),  ds_scores[ns][idx].item()])
        # Sort by score in descending order
        sorted_scores.sort(key=lambda x: x[2], reverse=True)
        sorted_order = []
        for item in sorted_scores:
            # item is [ns_type, pos, score]
            candidate = [item[0], preactivation_lubs_idx, item[1]]
            # candidate is [ns_type, lubs_layer_idx, pos]
            sorted_order.append(candidate)
        return sorted_order
