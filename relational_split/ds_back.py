import copy
import time
import torch
import gc
from common.network import LayerType
from relational_split.ds_handler import DS_handler
from common import Status
from relational_property.relational_analysis import relational_analysis_back, RelationalProperty
from dual.dual_network import get_relational_order


class DS_info:
    def __init__(self, ds_type, layer_idx, pos, split_value):

        self.ds_type = ds_type
        self.layer_idx = layer_idx  # int
        self.pos = pos  # int
        self.split_value = split_value


class DS(DS_handler):
    def __init__(self, log_file='log', DS_mode='DS_random_ABCD', split_limit=10, relational_prop=RelationalProperty.GLOBAL_ROBUSTNESS):

        self.IARb = None
        self.DS_mode = DS_mode
        self.DS_history = []
        self.DS_failed_list = []
        self.split_limit = split_limit
        self.log_file = log_file
        self.name = "DS"
        self.split_count = 0
        self.status = Status.UNKNOWN
        self.inp1_label = None
        self.inp2_label = None
        self.relational_output_dist = None
        self.relational_prop = relational_prop
        self.DS_candidates = {}
        self.selection_start_layer = 0
        if DS_mode in ['DS_random_Z']:
            self.get_ds_candidates = self.get_ds_candidates_random
        elif DS_mode in ['DS_dual_Z']:
            self.get_ds_candidates = self.get_ds_candidates_dual
        else:
            raise ValueError(f"Unknown DS mode: {DS_mode}. Expected 'DS_random_Z' or 'DS_dual_Z'.")

    def get_DS_info(self, ds_type, ds_position, IARb):
        if len(ds_position) != 2:
            raise ValueError(f"Expected ds_position to have length 2, but got {len(ds_position)}. "
                             f"This indicates that the DS position is not valid.")
        lubs_layer_idx, pos = ds_position[0], ds_position[1]
        if ds_type == 'DSZ':
            split_value = 0
        else:
            raise ValueError(f"Unknown DS type: {ds_type}. Expected 'DSZ'.")

        ds_info = DS_info(ds_type=ds_type, layer_idx=lubs_layer_idx, pos=pos, split_value=split_value)
        return ds_info

    def candidate_to_DS_info(self, candidate, IARb):
        # candidate = [ds_type, lubs_layer_idx, pos]
        ds_type = candidate[0]
        ds_position = candidate[1:]
        if ds_type in ('DSZ'):
            ds_info = self.get_DS_info(ds_type=ds_type, ds_position=ds_position, IARb=IARb)
        else:
            raise ValueError(f"Unknown DS type: {candidate[0]}. Expected 'DSZ'.")
        return ds_info

    def get_ds_candidates_dual(self, IARb, layer_idx, DN=None, RelAna=None):
        if RelAna is not None:
            target_dim = RelAna.inp1_correct_label
            C = torch.zeros_like(IARb.inp1_lbs[-1])
            C[target_dim] = 1
        else:
            C = torch.ones_like(IARb.inp1_lbs[-1])
        ds_order = get_relational_order(IARb.net, C, self.DS_mode, layer_idx, IARb.shapes, IARb.inp1_lbs, IARb.inp1_ubs,
                                        IARb.inp2_lbs, IARb.inp2_ubs, IARb.d_lbs, IARb.d_ubs)
        # ds_order = [[ds_type,lubs_layer_idx, pos], ...]
        return None, ds_order

    def get_ds_candidates_random(self, IARb, layer_idx, RelAna=None):
        """
        DS_random_Z: randomly selects from DSZ candidates
        """

        ds_order = self.get_ds_random_order(self.DS_mode, layer_idx, IARb.inp1_lbs, IARb.inp1_ubs, IARb.inp2_lbs, IARb.inp2_ubs,
                                            IARb.d_lbs, IARb.d_ubs)
        # ds_order = [[ds_type, lubs_layer_idx, pos], ...]
        return None, ds_order

    def run_iterative_DS_backend(self, IARb, RelAna, time_budget=1000, lp_analysis=False):
        """
        bfs: breadth-first search

        bfs_ds_list: list of DS instances to be explored
        bfs_relAna_list: list of RelationalAnalysis instances corresponding to bfs_ds_list

        DS_list: list of DS instances that have been verified or found adversarial examples
        DS_history: list of DS_info instances representing how splitting has been done
        """

        self.IARb = IARb
        bfs_ds_list = [self]
        DS_list = []
        self.status = Status.UNKNOWN
        if self.split_count >= self.split_limit:
            return Status.UNKNOWN, None, None

        time_start_grd = time.time()  # time tracking start
        split_count = 0
        bfs_res = None
        ds_order_selection = None
        first_relu_layer_skip = False  # skip the first relu layer, because splitting relu at the first layer by DSM and DSZ is same
        termination_condition = (len(bfs_ds_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        while not termination_condition:
            left_time_budget = time_budget - (time.time() - time_start_grd)
            bfs_res, next_bfs_list = self.bfs_loop(bfs_ds_list, left_time_budget, RelAna, first_relu_layer_skip, DS_list, split_count, ds_order_selection)
            if len(next_bfs_list) == 1 and next_bfs_list[0] == self:
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nNo further splits possible on Root problem. Terminating DS process.\n")
                break
            bfs_ds_list = next_bfs_list
            split_count += 1
            termination_condition = (len(bfs_ds_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        return bfs_res

    def bfs_loop(self, bfs_list, time_budget, RelAna, first_relu_layer_skip, DS_list, split_count, ds_order_selection):
        """
        for each bfs ds instance in the bfs list:
        1. backsubstitution based on the ds history
        2. candidate selection
        3. lp analysis
        """
        time_start_grd = time.time()
        ds_res_level = []  # to keep ds sets of this candidate excluding unreachable ones
        next_bfs_list = []
        for bfs_idx in range(len(bfs_list)):
            time_progress = time.time() - time_start_grd
            exe_flag = False
            if time_progress > time_budget:
                ds_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                next_bfs_list = []
                break
            time_bfs_start = time.time()

            # current DS instance
            bfs_ds = bfs_list[bfs_idx]
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## BFS DS instance: {bfs_ds.name}\n")

            # ---- update bounds by backsubstitution ----
            update_start_time = time.time()
            curr_iarb = copy.deepcopy(bfs_ds.IARb)
            if len(bfs_ds.DS_history) > 0 and curr_iarb.feasible_flag is not False:
                split_layer_list = [ds_info.layer_idx for ds_info in bfs_ds.DS_history]
                split_layer_list = list(set(split_layer_list))  # unique layer indices, e.g., [1,1,3,3,5] -> [1,3,5]
                curr_iarb = curr_iarb.update_bounds_IAR(ds_history=bfs_ds.DS_history, split_layer_list=split_layer_list)
                if curr_iarb is None:
                    # backsubstitution failed
                    bfs_ds.IARb.feasible_flag = False  # mark as infeasible
                    curr_iarb = bfs_ds.IARb

                # ---- debug ----
                if curr_iarb.feasible_flag is not False:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Backsubstitution after applying DS history:\n")
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
            for idx in range(bfs_ds.selection_start_layer, len(curr_iarb.net)):
                if exe_flag:
                    break  # proceed to next bfs ds instance
                layer = curr_iarb.net[idx]
                time_progress = time.time() - time_start_grd
                if time_progress > time_budget:
                    ds_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                    next_bfs_list = []
                    break
                if layer.type is LayerType.ReLU and first_relu_layer_skip is False:
                    if curr_iarb.feasible_flag is False and bfs_ds.DS_candidates != {}:
                        ds_candidates = bfs_ds.DS_candidates[idx]
                    else:
                        if ds_order_selection:
                            ds_order_selection, ds_candidates = self.get_ds_candidates(curr_iarb, idx, ds_order_selection)
                            bfs_ds.DS_candidates[idx] = ds_candidates
                        else:
                            ds_order_selection, ds_candidates = self.get_ds_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna)  # candidates: [[ds_type, lubs_layer_idx, pos], ...]
                            bfs_ds.DS_candidates[idx] = ds_candidates
                    if len(ds_candidates) == 0:
                        bfs_ds.selection_start_layer += 1
                        continue
                elif idx == len(curr_iarb.net) - 1:  # last layer
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo DS candidates found\n")
                    break
                else:
                    bfs_ds.selection_start_layer += 1
                    continue

                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\n### DS candidates at layer {idx}\n")
                if len(ds_candidates) == 0 or ds_candidates is None:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo DS candidates found\n")
                    continue
                else:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        for candidate in ds_candidates:
                            f.write(f"type: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")

                candidate_selection_time = time.time() - candidate_selection_start
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nTime for candidate selection: {candidate_selection_time:.2f} seconds\n")

                progress_time = time.time() - time_start_grd
                if progress_time > time_budget:
                    break

                # ---- split LP analysis ----
                lp_analysis_start = time.time()
                for candidate in ds_candidates:
                    progress_time = time.time() - time_start_grd
                    if progress_time > time_budget:
                        ds_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                        next_bfs_list = []
                        break
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Candidate\ntype: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")
                    ds_info = self.candidate_to_DS_info(candidate, bfs_ds.IARb)
                    # check whether the same split has been done
                    prev_layer_pos = [(h.layer_idx, h.pos) for h in bfs_ds.DS_history]
                    if (ds_info.layer_idx, ds_info.pos) in prev_layer_pos:
                        # try next candidate
                        continue
                    # check whether the same split has failed before
                    prev_failed_splits = [(h.layer_idx, h.pos, h.ds_type) for h in bfs_ds.DS_failed_list]
                    if (ds_info.layer_idx, ds_info.pos, ds_info.ds_type) in prev_failed_splits:
                        # try next candidate
                        continue
                    ds_info_1 = copy.deepcopy(ds_info)
                    ds_info_2 = copy.deepcopy(ds_info)
                    ds_info_1.ds_type = f"{ds_info.ds_type}1"
                    ds_info_2.ds_type = f"{ds_info.ds_type}2"
                    ds1 = copy.deepcopy(bfs_ds)
                    ds1.name = f"{bfs_ds.name}_{ds_info_1.ds_type}"
                    ds1.DS_history.append(ds_info_1)
                    ds2 = copy.deepcopy(bfs_ds)
                    ds2.name = f"{bfs_ds.name}_{ds_info_2.ds_type}"
                    ds2.DS_history.append(ds_info_2)
                    status1, status2 = self.perform_ds_1_2(ds1, ds2, RelAna, bfs_ds.IARb, ds_type="ABCD")
                    if status1 == Status.UNREACHABLE or status2 == Status.UNREACHABLE:
                        bfs_ds.DS_failed_list.append(ds_info)
                        # free up memory
                        del ds1
                        del ds2
                        gc.collect()
                        continue

                    ds1.IARb = curr_iarb
                    ds2.IARb = curr_iarb
                    exe_flag = True
                    for curr_ds, curr_status in zip([ds1, ds2], [status1, status2]):
                        if curr_status == Status.UNKNOWN:
                            next_bfs_list.append(curr_ds)
                            ds_res_level.append(curr_ds)
                        elif curr_status == Status.ADV_EXAMPLE:
                            DS_list.append(curr_ds)
                            return (curr_status, curr_ds.inp1_label, curr_ds.inp2_label), []
                        elif curr_status == Status.VERIFIED:
                            DS_list.append(curr_ds)
                            ds_res_level.append(curr_ds)
                        elif curr_status == Status.UNREACHABLE:
                            raise ValueError("This case should have been handled earlier.")
                        else:
                            raise ValueError(f"Unknown status: {curr_status}")
                    if exe_flag:
                        break
                if not exe_flag:
                    bfs_ds.selection_start_layer += 1
            if not exe_flag:
                return (Status.UNKNOWN, None, None), []  # we cannot reach satisfying the output specification

        # ---- show results of bfs loop ----
        bfs_time = time.time() - time_bfs_start
        if len(ds_res_level) > 0:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## Summary of splitting (split count: {split_count})\n")
                f.write(f"- Time for DS candidates: {bfs_time:.2f} seconds\n")
                for ds_ in ds_res_level:
                    f.write(f"{ds_.name}, status: {ds_.status}, split count: {ds_.split_count}, time: {bfs_time:.2f}\n")
                    for dim, dist in ds_.relational_output_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")

        status, inp1_out_label, inp2_out_label = self.collect_final_status(DS_list + ds_res_level)

        return (status, inp1_out_label, inp2_out_label), next_bfs_list

    def perform_ds_1_2(self, ds1, ds2, RelAna, IARb, ds_type):
        for curr_ds in [ds1, ds2]:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n### Relational analysis {ds_type} of {curr_ds.name}\n")
            ra_time_start = time.time()
            curr_status, inp1_label, inp2_label, rel_out_dist = relational_analysis_back(IARb=IARb, RelAna=RelAna, DS_history=curr_ds.DS_history,
                                                                                         log_file=self.log_file)
            if curr_status == Status.UNREACHABLE:
                return Status.UNREACHABLE, Status.UNREACHABLE
            curr_ds.split_count += 1
            curr_ds.relational_output_dist = rel_out_dist
            curr_ds.status = curr_status
            curr_ds.inp1_label = inp1_label
            curr_ds.inp2_label = inp2_label
            ra_time_end = time.time()
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n#### Relational analysis {ds_type} result of {curr_ds.name}\n")
                f.write(f"Status: {curr_status}\n")
                if rel_out_dist is not None:
                    for dim, dist in rel_out_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")
                f.write(f"time: {(ra_time_end - ra_time_start):.2f} seconds\n")
        return ds1.status, ds2.status
