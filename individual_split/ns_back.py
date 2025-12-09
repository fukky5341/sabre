import copy
import time
import torch
import gc
from common.network import LayerType
from individual_split.ns_handler import NS_handler
from common import Status
from relational_property.relational_analysis import relational_analysis_back, RelationalProperty
from dual.dual_network import get_relational_order_ns
from dual.dual_network_ind import get_relational_order_ns_ind


class NS_info:
    def __init__(self, ns_inp, layer_idx, pos, split_value):

        self.ns_type = None  # None for now, will be set as ("A1", "A2", "B1", "B2")
        self.ns_inp = ns_inp  # str: A or B, which indicates which individual network to split
        self.layer_idx = layer_idx  # int
        self.pos = pos  # int
        self.split_value = split_value


class NS(NS_handler):
    def __init__(self, log_file='log', NS_mode='NS_random', split_limit=10, relational_prop=RelationalProperty.GLOBAL_ROBUSTNESS):

        # self.IARb = IndividualAndRelationalBounds
        self.NS_mode = NS_mode
        self.NS_history = []
        self.next_candidate_input = 'A'  # 'A' or 'B'
        self.NS_failed_list = []
        self.NS_candidates = {}
        self.NS_candidates_A = {}
        self.NS_candidates_B = {}
        self.split_limit = split_limit
        self.log_file = log_file
        self.name = "NS"
        self.split_count = 0
        self.status = Status.UNKNOWN
        self.inp1_label = None
        self.inp2_label = None
        self.relational_output_dist = None  # {dim: [lb, ub], ...}
        self.relational_prop = relational_prop
        self.selection_start_layer = 0  # to speed up candidate selection
        if NS_mode == 'NS_random':
            self.get_ns_candidates = self.get_ns_candidates_random
        elif NS_mode == 'NS_dual':
            self.get_ns_candidates = self.get_ns_candidates_dual
        elif NS_mode == 'NS_dual_ind':
            self.get_ns_candidates = self.get_ns_candidates_dual_ind
        else:
            raise ValueError(f"Unknown NS_mode: {NS_mode}")

    def get_NS_info(self, ns_position):
        if len(ns_position) != 3:
            raise ValueError(f"Expected ns_position to have length 3, but got {len(ns_position)}. "
                             f"This indicates that the NS position is not valid.")

        ns_inp, lubs_layer_idx, pos = ns_position[0], ns_position[1], ns_position[2]
        split_value = 0

        ns_info = NS_info(ns_inp=ns_inp, layer_idx=lubs_layer_idx, pos=pos, split_value=split_value)
        return ns_info

    def candidate_to_NS_info(self, candidate):
        # candidate = [lubs_layer_idx, pos]
        ns_info = self.get_NS_info(ns_position=candidate)
        return ns_info

    # todo
    def get_ns_candidates_dual(self, IARb, layer_idx, DN=None, RelAna=None):
        if RelAna is not None:
            target_dim = RelAna.inp1_correct_label
            C = torch.zeros_like(IARb.inp1_lbs[-1])
            C[target_dim] = 1
        else:
            C = torch.ones_like(IARb.inp1_lbs[-1])
        ns_order = get_relational_order_ns(IARb.net, C, self.NS_mode, layer_idx, IARb.shapes, IARb.inp1_lbs, IARb.inp1_ubs,
                                           IARb.inp2_lbs, IARb.inp2_ubs, IARb.d_lbs, IARb.d_ubs)
        # ns_order = [[ns_inp,lubs_layer_idx, pos], ...]
        return None, ns_order

    def get_ns_candidates_dual_ind(self, IARb, layer_idx, DNI=None, RelAna=None, input_ab='A'):
        if RelAna is not None:
            target_dim = RelAna.inp1_correct_label
            C = torch.zeros_like(IARb.inp1_lbs[-1])
            C[target_dim] = 1
        else:
            C = torch.ones_like(IARb.inp1_lbs[-1])
        if input_ab == 'A':
            lbs = IARb.inp1_lbs
            ubs = IARb.inp1_ubs
        elif input_ab == 'B':
            lbs = IARb.inp2_lbs
            ubs = IARb.inp2_ubs
        else:
            raise ValueError(f"Unknown input_ab: {input_ab}")
        ns_order = get_relational_order_ns_ind(IARb.net, C, self.NS_mode, layer_idx, IARb.shapes, lbs, ubs)
        # ns_order = [[ns_inp, lubs_layer_idx, pos], ...]

        if input_ab == 'B':
            # change ns_inp from 'A' to 'B'
            for item in ns_order:
                item[0] = 'B'

        return None, ns_order

    def run_iterative_NS_backend(self, IARb, RelAna, time_budget=1000, lp_analysis=False):
        self.IARb = IARb
        bfs_dsns_list = [self]
        DSNS_list = []
        self.status = Status.UNKNOWN
        if self.split_count >= self.split_limit:
            return Status.UNKNOWN, None, None

        time_start_grd = time.time()  # time tracking start
        split_count = 0
        bfs_res = None
        dsns_order_selection = None
        first_relu_layer_skip = False  # skip the first relu layer, because splitting relu at the first layer by DSM and DSZ is same
        termination_condition = (len(bfs_dsns_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        while not termination_condition:
            left_time_budget = time_budget - (time.time() - time_start_grd)
            bfs_res, next_bfs_list = self.bfs_loop(bfs_dsns_list, left_time_budget, RelAna, first_relu_layer_skip, DSNS_list, split_count, dsns_order_selection)
            if len(next_bfs_list) == 1 and next_bfs_list[0] == self:
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nNo further splits possible on Root problem. Terminating DS process.\n")
                break
            bfs_dsns_list = next_bfs_list
            split_count += 1
            termination_condition = (len(bfs_dsns_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        return bfs_res

    def bfs_loop(self, bfs_list, time_budget, RelAna, first_relu_layer_skip, NS_list, split_count, ns_order_selection):
        """
        for each bfs ns instance in the bfs list:
        1. backsubstitution based on the ns history
        2. candidate selection
        3. lp analysis
        """
        time_start_grd = time.time()
        ns_res_level = []  # to keep ns sets of this candidate excluding unreachable ones
        next_bfs_list = []
        for bfs_idx in range(len(bfs_list)):
            time_progress = time.time() - time_start_grd
            exe_flag = False
            if time_progress > time_budget:
                ns_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                next_bfs_list = []
                break
            time_bfs_start = time.time()

            # current NS instance
            bfs_ns = bfs_list[bfs_idx]
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## BFS NS instance: {bfs_ns.name}\n")

            # ---- update bounds by backsubstitution ----
            update_start_time = time.time()
            curr_iarb = copy.deepcopy(bfs_ns.IARb)
            if len(bfs_ns.NS_history) > 0 and curr_iarb.feasible_flag is not False:
                split_layer_list = [ns_info.layer_idx for ns_info in bfs_ns.NS_history]
                split_layer_list = list(set(split_layer_list))  # unique layer indices, e.g., [1,1,3,3,5] -> [1,3,5]
                curr_iarb = curr_iarb.update_bounds_IAR(ns_history=bfs_ns.NS_history, split_layer_list=split_layer_list)
                if curr_iarb is None:
                    # backsubstitution failed
                    bfs_ns.IARb.feasible_flag = False  # mark as infeasible
                    curr_iarb = bfs_ns.IARb

                # ---- debug ----
                if curr_iarb.feasible_flag is not False:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Backsubstitution after applying NS history:\n")
                        for dim in range(len(curr_iarb.inp1_lbs[-1])):
                            f.write(f"{dim}: {curr_iarb.inp1_lbs[-1][dim].item():.7f}, {curr_iarb.inp1_ubs[-1][dim].item():.7f}, "
                                    f"{curr_iarb.inp2_lbs[-1][dim].item():.7f}, {curr_iarb.inp2_ubs[-1][dim].item():.7f}, "
                                    f"{curr_iarb.d_lbs[-1][dim].item():.7f}, {curr_iarb.d_ubs[-1][dim].item():.7f}\n")
                # ---- debug ----
            update_time = time.time() - update_start_time
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\nTime for backsubstitution: {update_time:.2f} seconds\n")

            # ---- candidate selection ----
            candidate_selection_start = time.time()
            start_layer = bfs_ns.selection_start_layer if self.NS_mode != 'NS_dual_ind' else 0
            for idx in range(start_layer, len(curr_iarb.net)):
                if exe_flag:
                    break  # proceed to next bfs ns instance
                layer = curr_iarb.net[idx]
                time_progress = time.time() - time_start_grd
                if time_progress > time_budget:
                    ns_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                    next_bfs_list = []
                    break
                if layer.type is LayerType.ReLU and first_relu_layer_skip is False:
                    if self.NS_mode == 'NS_dual':
                        if curr_iarb.feasible_flag is False and bfs_ns.NS_candidates != {}:
                            ns_candidates = bfs_ns.NS_candidates[idx]
                        else:
                            if ns_order_selection:
                                ns_order_selection, ns_candidates = self.get_ns_candidates(curr_iarb, idx, ns_order_selection)
                                bfs_ns.NS_candidates[idx] = ns_candidates
                            else:
                                ns_order_selection, ns_candidates = self.get_ns_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna)  # candidates: [[ns_type, lubs_layer_idx, pos], ...]
                                bfs_ns.NS_candidates[idx] = ns_candidates
                    elif self.NS_mode == 'NS_dual_ind':
                        if bfs_ns.next_candidate_input == 'A':
                            if curr_iarb.feasible_flag is False and bfs_ns.NS_candidates_A != {}:
                                ns_candidates = bfs_ns.NS_candidates_A[idx]
                            else:
                                if ns_order_selection:
                                    ns_order_selection, ns_candidates = self.get_ns_candidates(curr_iarb, idx, ns_order_selection, input_ab='A')
                                    bfs_ns.NS_candidates_A[idx] = ns_candidates
                                else:
                                    ns_order_selection, ns_candidates = self.get_ns_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna,
                                                                                               input_ab='A')  # candidates: [[ns_type, lubs_layer_idx, pos], ...]
                                    bfs_ns.NS_candidates_A[idx] = ns_candidates
                        elif bfs_ns.next_candidate_input == 'B':
                            if curr_iarb.feasible_flag is False and bfs_ns.NS_candidates_B != {}:
                                ns_candidates = bfs_ns.NS_candidates_B[idx]
                            else:
                                if ns_order_selection:
                                    ns_order_selection, ns_candidates = self.get_ns_candidates(curr_iarb, idx, ns_order_selection, input_ab='B')
                                    bfs_ns.NS_candidates_B[idx] = ns_candidates
                                else:
                                    ns_order_selection, ns_candidates = self.get_ns_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna,
                                                                                               input_ab='B')  # candidates: [[ns_type, lubs_layer_idx, pos], ...]
                                    bfs_ns.NS_candidates_B[idx] = ns_candidates
                        else:
                            raise ValueError(f"Unknown next_candidate_input: {bfs_ns.next_candidate_input}")
                elif idx == len(curr_iarb.net) - 1:  # last layer
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo NS candidates found\n")
                    break
                else:
                    bfs_ns.selection_start_layer += 1
                    continue

                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\n### NS candidates at layer {idx}\n")
                if len(ns_candidates) == 0 or ns_candidates is None:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo NS candidates found\n")
                    continue
                else:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        for candidate in ns_candidates:
                            f.write(f"type: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")

                candidate_selection_time = time.time() - candidate_selection_start
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nTime for candidate selection: {candidate_selection_time:.2f} seconds\n")

                progress_time = time.time() - time_start_grd
                if progress_time > time_budget:
                    break

                # ---- split LP analysis ----
                lp_analysis_start = time.time()
                exe_flag = False
                for candidate in ns_candidates:
                    progress_time = time.time() - time_start_grd
                    if progress_time > time_budget:
                        ns_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                        next_bfs_list = []
                        break
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Candidate\ntype: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")
                    ns_info = self.candidate_to_NS_info(candidate)
                    # check whether the same split has been done
                    prev_layer_pos = [(h.ns_type, h.layer_idx, h.pos) for h in bfs_ns.NS_history]
                    if (ns_info.ns_type, ns_info.layer_idx, ns_info.pos) in prev_layer_pos:
                        # try next candidate
                        continue
                    # check whether the same split has failed before
                    prev_failed_splits = [(h.ns_type, h.layer_idx, h.pos) for h in bfs_ns.NS_failed_list]
                    if (ns_info.ns_type, ns_info.layer_idx, ns_info.pos) in prev_failed_splits:
                        # try next candidate
                        continue
                    ns_info_1 = copy.deepcopy(ns_info)
                    ns_info_2 = copy.deepcopy(ns_info)
                    ns_info_1.ns_type = f"{ns_info.ns_inp}1"
                    ns_info_2.ns_type = f"{ns_info.ns_inp}2"
                    ns1 = copy.deepcopy(bfs_ns)
                    ns1.name = f"{bfs_ns.name}_{ns_info_1.ns_type}"
                    ns1.NS_history.append(ns_info_1)
                    ns2 = copy.deepcopy(bfs_ns)
                    ns2.name = f"{bfs_ns.name}_{ns_info_2.ns_type}"
                    ns2.NS_history.append(ns_info_2)
                    status1, status2 = self.perform_ns_1_2(ns1, ns2, RelAna, bfs_ns.IARb)
                    if status1 == Status.UNREACHABLE or status2 == Status.UNREACHABLE:
                        bfs_ns.NS_failed_list.append(ns_info)
                        # free up memory
                        del ns1
                        del ns2
                        gc.collect()
                        continue

                    ns1.IARb = curr_iarb
                    ns2.IARb = curr_iarb
                    exe_flag = True
                    if self.NS_mode == 'NS_dual_ind':
                        if bfs_ns.next_candidate_input == 'A':
                            ns1.next_candidate_input = 'B'
                            ns2.next_candidate_input = 'B'
                        else:
                            ns1.next_candidate_input = 'A'
                            ns2.next_candidate_input = 'A'
                    for curr_ns, curr_status in zip([ns1, ns2], [status1, status2]):
                        if curr_status == Status.UNKNOWN:
                            next_bfs_list.append(curr_ns)
                            ns_res_level.append(curr_ns)
                        elif curr_status == Status.ADV_EXAMPLE:
                            NS_list.append(curr_ns)
                            return (curr_status, curr_ns.inp1_label, curr_ns.inp2_label), []
                        elif curr_status == Status.VERIFIED:
                            NS_list.append(curr_ns)
                            ns_res_level.append(curr_ns)
                        elif curr_status == Status.UNREACHABLE:
                            raise ValueError("This case should have been handled earlier.")
                        else:
                            raise ValueError(f"Unknown status: {curr_status}")
                    if exe_flag:
                        break
                if not exe_flag:
                    bfs_ns.selection_start_layer += 1
            if not exe_flag:
                return (Status.UNKNOWN, None, None), []  # we cannot reach satisfying the output specification

        # ---- show results of bfs loop ----
        bfs_time = time.time() - time_bfs_start
        if len(ns_res_level) > 0:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## Summary of splitting at layer (split count: {split_count})\n")
                f.write(f"- Time for NS candidates: {bfs_time:.2f} seconds\n")
                for ns_ in ns_res_level:
                    f.write(f"{ns_.name}, status: {ns_.status}, split count: {ns_.split_count}, time: {bfs_time:.2f}\n")
                    for dim, dist in ns_.relational_output_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")

        status, inp1_out_label, inp2_out_label = self.collect_final_status(NS_list + ns_res_level)

        return (status, inp1_out_label, inp2_out_label), next_bfs_list

    def perform_ns_1_2(self, ns1, ns2, RelAna, IARb):
        for curr_ns in [ns1, ns2]:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## Relational analysis of {curr_ns.name}\n")
            ra_time_start = time.time()
            curr_status, inp1_label, inp2_label, rel_out_dist = relational_analysis_back(IARb=IARb, RelAna=RelAna, NS_history=curr_ns.NS_history,
                                                                                         log_file=self.log_file)
            if curr_status == Status.UNREACHABLE:
                return Status.UNREACHABLE, Status.UNREACHABLE
            curr_ns.split_count += 1
            curr_ns.relational_output_dist = rel_out_dist
            curr_ns.status = curr_status
            curr_ns.inp1_label = inp1_label
            curr_ns.inp2_label = inp2_label
            ra_time_end = time.time()
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n### Relational analysis result of {curr_ns.name}\n")
                f.write(f"Status: {curr_status}\n")
                if rel_out_dist is not None:
                    for dim, dist in rel_out_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")
                f.write(f"time: {(ra_time_end - ra_time_start):.2f} seconds\n")
        return ns1.status, ns2.status
