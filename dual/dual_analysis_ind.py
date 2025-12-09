import torch
import torch.nn as nn
import torch.nn.functional as F
from common.network import LayerType


class PreactivationBounds:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.positive = (lb >= 0) & (ub > 0)
        self.negative = (ub <= 0)
        self.unstable = ~(self.positive) & ~(self.negative)


class DualAnalysis_Ind:
    def __init__(self):
        pass

    def estimate_individual_impact(self, dualnet_1, dualnet_2, layer_idx, lbs, ubs):
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
            A^t bias <-- index of A is idx
        - interception term
            lambda*B <-- index of B is idx + 1

        return ns_order: [[ns_inp, lubs_layer_idx, pos], ...], e.g., [['A', 3, 10], ['B', 3, 10], ...]
        """

        idx = layer_idx
        split_scores_list = []
        for dual_network in [dualnet_1, dualnet_2]:
            split_scores, unstable_mask = self.get_ns_merged_scores(dual_network.dual_net[idx-1].bias, idx, lbs[idx], ubs[idx],
                                                                    dual_network.As[::-1][idx+1], dual_network.As[::-1][idx])
            if split_scores is None:
                return []
            split_scores_list.append(split_scores)

        # combine the two split scores
        combined_score = split_scores_list[0] + split_scores_list[1]

        # arrange in descending order
        ns_order = self.arrange_in_descending_order_ns(unstable_mask, combined_score, idx)
        # ns_order = [[lubs_layer_idx, pos], ...]

        return ns_order  # ns_order: [[ns_inp, lubs_layer_idx, pos], ...]

    def get_ns_merged_scores(self, bias, preactivation_lubs_idx, lb, ub, curr_v, next_v):
        """
        split method:
            - zero

        procedure:
        - get unstable relu
        - old score
        - new score
            score = old{A^t bias} - new{A^t bias} + new{ulB/(u-l)} - old{ulB/(u-l)}
                = old{A^t bias} - old{ulB/(u-l)} - new{A^t bias}
                (because, new{ulB/(u-l)} is 0)
        """
        PB = PreactivationBounds(lb, ub)
        if not PB.unstable.any():
            return None, None
        # old score
        old_score = self.get_old_score(bias, PB.unstable, PB, curr_v, next_v)
        # split score
        ns_scores = self.get_split_scores_ns(old_score, bias, PB, PB.unstable, curr_v)
        # arrange in descending order

        return ns_scores, PB.unstable

    def get_y_intercept_term_score(self, PB, curr_v):
        y_score = torch.zeros_like(curr_v)

        curr_v_positive = torch.clamp(curr_v, min=0)  # e.g., curr_v = [0.5, -0.5, 0.3, -0.2], then curr_v_positive = [0.5, 0, 0.3, 0]
        if PB.unstable.any():
            temp_c = (PB.ub*PB.lb) / (PB.ub - PB.lb + 1e-15)
            y_score = torch.where(PB.unstable, temp_c*curr_v_positive, y_score)

        return y_score

    def get_old_score(self, bias, common_condition, PB, curr_v, next_v):
        """
        old score: A^t bias - ulB/(u-l)
        - A^t bias <-- A: next_v
        - ulB/(u-l) <-- B: curr_v
        """
        # bias term
        bias_old_score = torch.where(common_condition, next_v * bias, torch.zeros_like(bias))

        # y-intercept term
        y_old_score = self.get_y_intercept_term_score(PB, curr_v)  # already filtered by unstable

        old_score = bias_old_score - y_old_score
        return old_score

    def get_split_scores_ns(self, old_score, bias, PB, ns_condition, curr_v):
        """
        curr_v is B

        ns: [l, u] --> (1)[l, 0], (2)[0, u]

        new score: - updated_A*bias = - (updated_lam * curr_v)*bias
        updated_lam can be 0 or 1

        1. bounds after splitting
        2. update conditions
        3. get score for each child
        4. merge scores
        """
        new_score = torch.where(ns_condition, -curr_v * bias, torch.zeros_like(bias))

        # merge scores
        split_score = new_score + torch.where(ns_condition, old_score, torch.zeros_like(old_score))

        return split_score

    def arrange_in_descending_order_ns(self, unstable_mask, ds_scores, preactivation_lubs_idx):
        """
        Arrange the scores in descending order.
        ds_scores consists of scores for each split condition.
        return: [['split_condition', score, position], ...]
        """
        sorted_scores = []
        active_idx = torch.where(unstable_mask)[0]
        if active_idx.numel() > 0:
            for idx in active_idx:
                sorted_scores.append([idx.item(),  ds_scores[idx].item()])  # [pos, score]
        # Sort by score in descending order
        sorted_scores.sort(key=lambda x: x[1], reverse=True)

        sorted_order = []
        for item in sorted_scores:
            sorted_order.append(['A', preactivation_lubs_idx, item[0]])  # [ns_inp, lubs_layer_idx, pos]
            # sorted_order.append(['B', preactivation_lubs_idx, item[0]])  # [ns_inp, lubs_layer_idx, pos]

        return sorted_order  # [[ns_inp, lubs_layer_idx, pos], ...]
