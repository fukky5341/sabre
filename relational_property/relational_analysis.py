from common import Status
import numpy as np
from common.network import LayerType
import copy
import gurobipy as gp
import torch
import torch.nn.functional as F
from enum import Enum
import gc


class RelationalProperty(Enum):
    GLOBAL_ROBUSTNESS = 1


def relational_analysis_back(IARb, RelAna, log_file="log", DS_history=None, NS_history=None):
    if RelAna.relational_prop is None:
        raise ValueError("Relational property is not set in RelAna. Please set it before running the analysis.")
    if DS_history is not None and NS_history is not None:
        raise ValueError("Both DS_history and NS_history are provided. Only one must be provided.")
    if RelAna.relational_prop == RelationalProperty.GLOBAL_ROBUSTNESS:
        status, delta_out_dist = RelAna.run_global_robustness_analysis(
            IARb=IARb, DS_history=DS_history, NS_history=NS_history, log_file=log_file)
        print(f"Global robustness analysis done")
        return status, None, None, delta_out_dist
    else:
        raise ValueError(f"Unknown relational property: {RelAna.relational_prop}. "
                         f"Expected 'GlobalRobustness'.")


class RelationalAnalysis:
    def __init__(self, relational_prop=None, relational_transformer='DiffPoly', lp_analysis=False, global_target=False, inp1_correct_label=None, inp2_correct_label=None, threshold=None, log_file="log"):
        self.relational_prop = relational_prop
        self.lp_analysis = lp_analysis
        self.global_target = global_target
        self.status = Status.UNKNOWN
        self.inp1_correct_label = inp1_correct_label
        self.inp2_correct_label = inp2_correct_label
        self.inp1_amb_labels = None
        self.inp2_amb_labels = None
        self.grb_inp1_vars = []
        self.grb_inp2_vars = []
        self.grb_delta_vars = []
        self.grb_model = None
        self.relu_constrs = {}  # (layer_idx, neuron_idx) -> [list of constraint names] note: layer_idx starts from 0 with input layer
        self.tolerance = 1e-4
        self.log_file = log_file
        self.ds_name = 'DS'
        self.relational_transformer = relational_transformer  # 'DiffPoly', 'ITNE'
        self.threshold = threshold
        #! check whether the added parameters are necessary to duplicate

    def duplicate(self, RelAna):
        self.relational_prop = RelAna.relational_prop
        self.lp_analysis = RelAna.lp_analysis
        self.global_target = RelAna.global_target
        self.inp1_correct_label = RelAna.inp1_correct_label
        self.inp2_correct_label = RelAna.inp2_correct_label
        self.inp1_amb_labels = copy.deepcopy(RelAna.inp1_amb_labels)
        self.inp2_amb_labels = copy.deepcopy(RelAna.inp2_amb_labels)
        self.relu_constrs = copy.deepcopy(RelAna.relu_constrs)
        self.tolerance = RelAna.tolerance
        self.copy_grb_model(RelAna)
        self.log_file = RelAna.log_file
        self.ds_name = RelAna.ds_name
        self.relational_transformer = RelAna.relational_transformer
        self.threshold = RelAna.threshold

    def duplicate_light(self, RelAna):
        self.relational_prop = RelAna.relational_prop
        self.lp_analysis = RelAna.lp_analysis
        self.global_target = RelAna.global_target
        self.inp1_correct_label = RelAna.inp1_correct_label
        self.inp2_correct_label = RelAna.inp2_correct_label
        self.inp1_amb_labels = copy.deepcopy(RelAna.inp1_amb_labels)
        self.inp2_amb_labels = copy.deepcopy(RelAna.inp2_amb_labels)
        self.tolerance = RelAna.tolerance
        self.log_file = RelAna.log_file
        self.ds_name = RelAna.ds_name
        self.relational_transformer = RelAna.relational_transformer
        self.threshold = RelAna.threshold

    def copy_grb_model(self, RelAna):
        if RelAna.grb_model is None:
            self.grb_model = None
        else:
            new_model = RelAna.grb_model.copy()
            new_model.update()

            # # debug
            # print(type(RelAna.grb_inp1_vars))
            # print(type(RelAna.grb_inp1_vars[0]))
            # print(type(RelAna.grb_inp1_vars[1][0]))

            def recover_mvar(old_mvar):
                var_names = [v.VarName for v in old_mvar.tolist()]
                return gp.MVar([new_model.getVarByName(name) for name in var_names])

            # update MVar lists
            self.grb_inp1_vars = [recover_mvar(mvar) for mvar in RelAna.grb_inp1_vars]
            self.grb_inp2_vars = [recover_mvar(mvar) for mvar in RelAna.grb_inp2_vars]
            self.grb_delta_vars = [recover_mvar(mvar) for mvar in RelAna.grb_delta_vars]
            new_model.update()

            self.grb_model = new_model

    def trash_used_relAna(self, temp_relAna):
        if temp_relAna is None:
            return
        if getattr(temp_relAna, "grb_model", None) is not None:
            temp_relAna.grb_model.dispose()
            temp_relAna.grb_model = None
        for attr in ("grb_inp1_vars", "grb_inp2_vars", "grb_delta_vars"):
            if getattr(temp_relAna, attr, None) is not None:
                setattr(temp_relAna, attr, [])
        del temp_relAna
        gc.collect()
        return

    def create_lp_model(self, net, shapes, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs, name=""):

        name = f"_{name}" if name else ""
        model = gp.Model(f"GlobalRobustness{name}")
        model.setParam('OutputFlag', 0)

        # $ Input layer

        lb1 = inp1_lbs[0].cpu().numpy()
        ub1 = inp1_ubs[0].cpu().numpy()
        lb2 = inp2_lbs[0].cpu().numpy()
        ub2 = inp2_ubs[0].cpu().numpy()
        lb_d = d_lbs[0].cpu().numpy()
        ub_d = d_ubs[0].cpu().numpy()

        n_vars = lb1.shape[0]
        inp1 = model.addMVar(n_vars, lb=lb1, ub=ub1, name="inp1_inp")
        inp2 = model.addMVar(n_vars, lb=lb2, ub=ub2, name="inp2_inp")
        delta = model.addMVar(n_vars, lb=lb_d, ub=ub_d, name="delta_inp")

        model.addConstr(delta == inp1 - inp2)

        model.update()
        self.grb_inp1_vars.append(inp1)
        self.grb_inp2_vars.append(inp2)
        self.grb_delta_vars.append(delta)

        # $ Hidden layers
        for layer_idx, layer in enumerate(net):  # 0, 1, 2, ..., L (layer_idx is the same as (lubs_idx-1))
            # Assuming the network has alternate affine and activation layer.
            linear_layer_index = layer_idx // 2
            # $ linear
            if layer.type is LayerType.Linear:
                W = layer.weight.detach().cpu().numpy()  # shape: (out_features, in_features)
                b = layer.bias.detach().cpu().numpy()     # shape: (out_features,)
                out_dim = W.shape[0]

                inp1_prev = self.grb_inp1_vars[-1]  # MVar
                inp2_prev = self.grb_inp2_vars[-1]  # MVar

                lb1 = inp1_lbs[layer_idx+1].cpu().numpy()
                ub1 = inp1_ubs[layer_idx+1].cpu().numpy()
                lb2 = inp2_lbs[layer_idx+1].cpu().numpy()
                ub2 = inp2_ubs[layer_idx+1].cpu().numpy()
                lb_d = d_lbs[layer_idx+1].cpu().numpy() - self.tolerance
                ub_d = d_ubs[layer_idx+1].cpu().numpy() + self.tolerance

                x1_out = model.addMVar(out_dim, lb=lb1, ub=ub1, name=f"inp1_lnr_{layer_idx+1}")
                x2_out = model.addMVar(out_dim, lb=lb2, ub=ub2, name=f"inp2_lnr_{layer_idx+1}")
                d_out = model.addMVar(out_dim, lb=lb_d, ub=ub_d, name=f"delta_lnr_{layer_idx+1}")

                model.addConstr(x1_out == W @ inp1_prev + b)
                model.addConstr(x2_out == W @ inp2_prev + b)
                model.addConstr(d_out == x1_out - x2_out)

                self.grb_inp1_vars.append(x1_out)
                self.grb_inp2_vars.append(x2_out)
                self.grb_delta_vars.append(d_out)

                model.update()
            # $ conv
            elif layer.type is LayerType.Conv2D:
                W = layer.weight.detach().cpu().numpy()  # shape: (C_out, C_in, kH, kW)
                B = layer.bias.detach().cpu().numpy()    # shape: (C_out,)
                stride_h, stride_w = layer.stride
                pad_h, pad_w = layer.padding
                dil_h, dil_w = layer.dilation

                C_in, H_in, W_in = preconv_shape = shapes[linear_layer_index]
                C_out, H_out, W_out = postconv_shape = shapes[linear_layer_index + 1]

                # def get_flat_index(c, h, w):
                #     return c * H_in * W_in + h * W_in + w

                total_out = C_out * H_out * W_out
                lb1 = inp1_lbs[layer_idx+1].cpu().numpy()
                ub1 = inp1_ubs[layer_idx+1].cpu().numpy()
                lb2 = inp2_lbs[layer_idx+1].cpu().numpy()
                ub2 = inp2_ubs[layer_idx+1].cpu().numpy()
                lb_d = d_lbs[layer_idx+1].cpu().numpy() - self.tolerance
                ub_d = d_ubs[layer_idx+1].cpu().numpy() + self.tolerance

                x1_out = model.addMVar(total_out, lb=lb1, ub=ub1, name=f"inp1_conv_{layer_idx+1}")
                x2_out = model.addMVar(total_out, lb=lb2, ub=ub2, name=f"inp2_conv_{layer_idx+1}")
                d_out = model.addMVar(total_out, lb=lb_d, ub=ub_d, name=f"delta_conv_{layer_idx+1}")

                inp1_flat = self.grb_inp1_vars[-1].tolist()
                assert len(inp1_flat) == C_in * H_in * W_in, f"size inconsistency: {len(inp1_flat)} != {C_in}x{H_in}x{W_in}"
                inp1_vars = np.array(inp1_flat).reshape(C_in, H_in, W_in)
                inp2_flat = self.grb_inp2_vars[-1].tolist()
                assert len(inp2_flat) == C_in * H_in * W_in, f"size inconsistency: {len(inp2_flat)} != {C_in}x{H_in}x{W_in}"
                inp2_vars = np.array(inp2_flat).reshape(C_in, H_in, W_in)

                for co in range(C_out):
                    for i in range(H_out):
                        for j in range(W_out):
                            out_idx = co * H_out * W_out + i * W_out + j

                            in_i = i * stride_h - pad_h
                            in_j = j * stride_w - pad_w
                            expr1_vars, expr2_vars, coeffs = [], [], []

                            for ci in range(C_in):
                                for kh in range(W.shape[2]):
                                    for kw in range(W.shape[3]):
                                        ih = in_i + kh * dil_h
                                        iw = in_j + kw * dil_w
                                        if 0 <= ih < H_in and 0 <= iw < W_in:
                                            expr1_vars.append(inp1_vars[ci, ih, iw])
                                            expr2_vars.append(inp2_vars[ci, ih, iw])
                                            coeffs.append(W[co, ci, kh, kw])

                            expr1 = gp.LinExpr(coeffs, expr1_vars)
                            expr1.addConstant(B[co])
                            expr2 = gp.LinExpr(coeffs, expr2_vars)
                            expr2.addConstant(B[co])

                            model.addConstr(x1_out[out_idx] == expr1)
                            model.addConstr(x2_out[out_idx] == expr2)
                            model.addConstr(d_out[out_idx] == x1_out[out_idx] - x2_out[out_idx])

                self.grb_inp1_vars.append(x1_out)
                self.grb_inp2_vars.append(x2_out)
                self.grb_delta_vars.append(d_out)

                model.update()
            # $ ReLU
            elif layer.type is LayerType.ReLU:
                # add vars
                n = inp1_lbs[layer_idx+1].shape[0]

                lb1 = inp1_lbs[layer_idx+1].cpu().numpy()
                clamp_lb1 = np.maximum(lb1, 0)
                ub1 = inp1_ubs[layer_idx+1].cpu().numpy()
                clamp_ub1 = np.maximum(ub1, 0)
                lb2 = inp2_lbs[layer_idx+1].cpu().numpy()
                clamp_lb2 = np.maximum(lb2, 0)
                ub2 = inp2_ubs[layer_idx+1].cpu().numpy()
                clamp_ub2 = np.maximum(ub2, 0)
                lb_d = d_lbs[layer_idx+1].cpu().numpy() - self.tolerance
                ub_d = d_ubs[layer_idx+1].cpu().numpy() + self.tolerance

                relu1 = model.addMVar(n, lb=clamp_lb1, ub=clamp_ub1, name=f"inp1_relu_{layer_idx+1}")
                relu2 = model.addMVar(n, lb=clamp_lb2, ub=clamp_ub2, name=f"inp2_relu_{layer_idx+1}")
                d_out = model.addMVar(n, lb=lb_d, ub=ub_d, name=f"delta_relu_{layer_idx+1}")

                # delta constraints
                model.addConstr(d_out == relu1 - relu2)

                # (lubs, layer_idx) [inp(0,none), layer0(1,0), layer1(2,1), ...]
                # we want to refer to the bounds of the previous layer
                # i.e., if current layer_idx is 2, we want to refer to lubs[2] (<-- layer_idx is 1)
                l1s = inp1_lbs[layer_idx].cpu().numpy()
                u1s = inp1_ubs[layer_idx].cpu().numpy()
                l2s = inp2_lbs[layer_idx].cpu().numpy()
                u2s = inp2_ubs[layer_idx].cpu().numpy()
                lds = d_lbs[layer_idx].cpu().numpy()
                uds = d_ubs[layer_idx].cpu().numpy()

                # vars at the previous layer
                relu1_in = self.grb_inp1_vars[-1]
                relu2_in = self.grb_inp2_vars[-1]
                d_in = self.grb_delta_vars[-1]

                for i in range(self.grb_inp1_vars[-1].shape[0]):
                    const_list = []

                    l1 = l1s[i]
                    u1 = u1s[i]
                    l2 = l2s[i]
                    u2 = u2s[i]
                    d_l = lds[i]
                    d_u = uds[i]

                    inp1_positive = ((l1 >= 0) & (u1 > 0))
                    inp1_negative = (u1 <= 0)
                    inp1_unsettled = (~inp1_positive & ~inp1_negative)
                    inp2_positive = ((l2 >= 0) & (u2 > 0))
                    inp2_negative = (u2 <= 0)
                    inp2_unsettled = (~inp2_positive & ~inp2_negative)
                    delta_positive = ((d_l >= 0) & (d_u > 0))
                    delta_negative = (d_u <= 0)
                    delta_unsettled = (~delta_positive & ~delta_negative)

                    # --- ReLU approximation for z1 ---
                    if inp1_positive:
                        model.addConstr(relu1[i] == relu1_in[i], name=f"const_relu1_{layer_idx+1}_{i}")  # todo: compare with constraints of lower and upper bounds
                        # pos_const_1 = model.addConstr(relu1[i] == relu1_in[i], name=f"const_relu1_{layer_idx}_{i}")  # todo: compare with constraints of lower and upper bounds
                        const_list.append(f"const_relu1_{layer_idx+1}_{i}")
                    elif inp1_negative:
                        model.addConstr(relu1[i] == 0, name=f"const_relu1_{layer_idx+1}_{i}")
                        # neg_const_1 = model.addConstr(relu1[i] == 0, name=f"const_relu1_{layer_idx}_{i}")
                        const_list.append(f"const_relu1_{layer_idx+1}_{i}")
                    else:
                        model.addConstr(relu1[i] >= 0, name=f"const_relu1_lb0_{layer_idx+1}_{i}")
                        model.addConstr(relu1[i] >= relu1_in[i], name=f"const_relu1_lb1_{layer_idx+1}_{i}")
                        slope = u1 / (u1 - l1 + 1e-6)
                        intercept = -l1 * slope
                        model.addConstr(relu1[i] <= slope * relu1_in[i] + intercept, name=f"const_relu1_ub_{layer_idx+1}_{i}")
                        const_list.extend([f"const_relu1_lb0_{layer_idx+1}_{i}", f"const_relu1_lb1_{layer_idx+1}_{i}", f"const_relu1_ub_{layer_idx+1}_{i}"])

                    # --- ReLU approximation for z2 ---
                    if inp2_positive:
                        model.addConstr(relu2[i] == relu2_in[i], name=f"const_relu2_{layer_idx+1}_{i}")
                        const_list.append(f"const_relu2_{layer_idx+1}_{i}")
                    elif inp2_negative:
                        model.addConstr(relu2[i] == 0, name=f"const_relu2_{layer_idx+1}_{i}")
                        const_list.append(f"const_relu2_{layer_idx+1}_{i}")
                    else:
                        model.addConstr(relu2[i] >= 0, name=f"const_relu2_lb0_{layer_idx+1}_{i}")
                        model.addConstr(relu2[i] >= relu2_in[i], name=f"const_relu2_lb1_{layer_idx+1}_{i}")
                        slope = u2 / (u2 - l2 + 1e-6)
                        intercept = -l2 * slope
                        model.addConstr(relu2[i] <= slope * relu2_in[i] + intercept, name=f"const_relu2_ub_{layer_idx+1}_{i}")
                        const_list.extend([f"const_relu2_lb0_{layer_idx+1}_{i}", f"const_relu2_lb1_{layer_idx+1}_{i}", f"const_relu2_ub_{layer_idx+1}_{i}"])

                    # --- Delta for ReLU outputs ---
                    """
                    we add further constraints as below based on RaVeN LP
                    I think there is a room for improvement.
                    """
                    # 1.
                    if inp1_unsettled & inp2_negative & delta_positive:
                        model.addConstr(d_out[i] <= d_in[i], name=f"const_delta_{layer_idx+1}_{i}")
                        const_list.append(f"const_delta_{layer_idx+1}_{i}")
                    # 2.
                    elif inp1_unsettled & inp2_positive:
                        model.addConstr(d_out[i] >= d_in[i], name=f"const_delta_{layer_idx+1}_{i}")
                        const_list.append(f"const_delta_{layer_idx+1}_{i}")
                    # 3.
                    elif inp1_negative & inp2_unsettled & delta_negative:
                        model.addConstr(d_out[i] >= d_in[i], name=f"const_delta_{layer_idx+1}_{i}")
                        const_list.append(f"const_delta_{layer_idx+1}_{i}")
                    # 4.
                    elif inp1_positive & inp2_unsettled:
                        model.addConstr(d_out[i] <= d_in[i], name=f"const_delta_{layer_idx+1}_{i}")
                        const_list.append(f"const_delta_{layer_idx+1}_{i}")
                    # 5.
                    elif inp1_unsettled & inp2_unsettled & delta_positive:
                        model.addConstr(d_out[i] <= d_in[i], name=f"const_delta_ub_{layer_idx+1}_{i}")
                        model.addConstr(d_out[i] >= 0, name=f"const_delta_lb_{layer_idx+1}_{i}")
                        const_list.extend([f"const_delta_ub_{layer_idx+1}_{i}", f"const_delta_lb_{layer_idx+1}_{i}"])
                    # 6.
                    elif inp1_unsettled & inp2_unsettled & delta_negative:
                        model.addConstr(d_out[i] >= d_in[i], name=f"const_delta_lb_{layer_idx+1}_{i}")
                        model.addConstr(d_out[i] <= 0, name=f"const_delta_ub_{layer_idx+1}_{i}")
                        const_list.extend([f"const_delta_lb_{layer_idx+1}_{i}", f"const_delta_ub_{layer_idx+1}_{i}"])
                    # 7.
                    elif inp1_unsettled & inp2_unsettled & delta_unsettled:
                        slope_ub = d_u / (d_u - d_l + 1e-6)
                        intercept_ub = -d_l * slope_ub
                        slope_lb = -d_l / (d_u - d_l + 1e-6)
                        intercept_lb = -d_u * slope_lb

                        model.addConstr(d_out[i] <= slope_ub * d_in[i] + intercept_ub, name=f"const_delta_ub_{layer_idx+1}_{i}")
                        model.addConstr(d_out[i] >= slope_lb * d_in[i] + intercept_lb, name=f"const_delta_lb_{layer_idx+1}_{i}")
                        const_list.extend([f"const_delta_ub_{layer_idx+1}_{i}", f"const_delta_lb_{layer_idx+1}_{i}"])

                    self.relu_constrs[(layer_idx+1, i)] = const_list  # layer_idx starts from 0 with input layer

                model.update()

                self.grb_inp1_vars.append(relu1)
                self.grb_inp2_vars.append(relu2)
                self.grb_delta_vars.append(d_out)

        return model

    def update_lp_model(self, model, DS_history=None, NS_history=None, inp1_lbs=None, inp1_ubs=None, inp2_lbs=None, inp2_ubs=None, d_lbs=None, d_ubs=None):

        # $ add split constraints
        if DS_history is not None and NS_history is not None:
            raise ValueError("Both DS_history and NS_history are provided. Only one must be provided.")

        elif DS_history is not None:
            for ds_info in DS_history:
                ds_type = ds_info.ds_type
                if "1" in ds_type:
                    self.add_ds_1_grb_constraints(model=model, DS_info=ds_info, inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs,
                                                  inp2_lbs=inp2_lbs, inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                elif "2" in ds_type:
                    self.add_ds_2_grb_constraints(model=model, DS_info=ds_info, inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs,
                                                  inp2_lbs=inp2_lbs, inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                else:
                    raise ValueError(f"Unsupported DS type: {ds_type}")
        elif NS_history is not None:
            for ns_info in NS_history:
                ns_type = ns_info.ns_type
                if "A1" in ns_type:
                    self.add_ns_A1_grb_constraints(model=model, NS_info=ns_info, inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs,
                                                   inp2_lbs=inp2_lbs, inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                elif "A2" in ns_type:
                    self.add_ns_A2_grb_constraints(model=model, NS_info=ns_info, inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs,
                                                   inp2_lbs=inp2_lbs, inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                elif "B1" in ns_type:
                    self.add_ns_B1_grb_constraints(model=model, NS_info=ns_info, inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs,
                                                   inp2_lbs=inp2_lbs, inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                elif "B2" in ns_type:
                    self.add_ns_B2_grb_constraints(model=model, NS_info=ns_info, inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs,
                                                   inp2_lbs=inp2_lbs, inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                else:
                    raise ValueError(f"Unsupported NS type: {ns_type}")
        else:
            raise ValueError("Both DS_history and NS_history are None. At least one must be provided.")

        return model

    def minimize_lp(self, model, target, disable_presolve=False, log_file=None):
        if disable_presolve:
            model.setParam("Presolve", 0)
        if log_file is not None:
            model.setParam("LogFile", log_file)

        model.setObjective(target, gp.GRB.MINIMIZE)
        model.optimize()

        status = model.status
        if status == gp.GRB.INF_OR_UNBD:
            model.setParam("DualReductions", 0)
            model.optimize()
            status = model.status

        if status == gp.GRB.OPTIMAL:
            return model.ObjVal
        elif status == gp.GRB.INFEASIBLE:
            print(f"Model infeasible (status={status})")
            # model.computeIIS()
            # model.write("model.ilp")
            model.write("model.lp")
            return None
        elif status == gp.GRB.UNBOUNDED:
            try:
                vars_ = model.getVars()
                ray = model.getAttr(gp.GRB.Attr.UnbdRay, vars_)
                nz = [(v.VarName, val) for v, val in zip(vars_, ray) if abs(val) > 1e-12]
                print("Unbounded ray (nonzeros):", nz[:20], " ...")  # 長い場合は一部表示
            except gp.GurobiError:
                pass
            model.write("model.lp")
            raise RuntimeError(f"LP is unbounded, check model.ilp for details. (status={model.status})")
        else:
            model.write("model.lp")
            raise RuntimeError(f"LP optimization failed with status {model.status}")

    def maximize_lp(self, model, target, disable_presolve=False, log_file=None):
        if disable_presolve:
            model.setParam("Presolve", 0)
        if log_file is not None:
            model.setParam("LogFile", log_file)

        model.setObjective(target, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.status == gp.GRB.OPTIMAL:
            return model.ObjVal
        elif model.status == gp.GRB.INFEASIBLE:
            print(f"Model infeasible (status={model.status})")
            model.computeIIS()
            model.write("model.ilp")
            model.write("model.lp")
            return None
        elif model.status == gp.GRB.UNBOUNDED:
            print(f"Model unbounded (status={model.status})")
            model.computeIIS()
            model.write("model.ilp")
            model.write("model.lp")
            raise RuntimeError(f"LP is unbounded, check model.ilp for details. (status={model.status})")
        else:
            raise RuntimeError(f"LP optimization failed with status {model.status}")

    def remove_model_constraints(self, model, layer_idx, pos, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub):
        # note: layer_idx indicates the preactivation layer of lubs index
        relu_layer_idx = layer_idx+1  # lubs index
        constr_names = self.relu_constrs[(relu_layer_idx, pos)]
        # collect constraint names
        for name in constr_names:
            constr = model.getConstrByName(name)
            if constr is not None:
                model.remove(constr)
        model.update()

    def update_model_constraints(self, model, layer_idx, pos, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub):
        # note: layer_idx indicates the preactivation layer of lubs index
        relu_layer_idx = layer_idx+1  # lubs index
        pre_relu1 = self.grb_inp1_vars[layer_idx][pos]
        pre_relu2 = self.grb_inp2_vars[layer_idx][pos]
        pre_delta = self.grb_delta_vars[layer_idx][pos]
        relu1 = self.grb_inp1_vars[relu_layer_idx][pos]
        relu2 = self.grb_inp2_vars[relu_layer_idx][pos]
        delta = self.grb_delta_vars[relu_layer_idx][pos]
        # add new constraints
        inp1_positive = ((inp1_lb < 0) & (inp1_ub > 0))
        inp1_negative = (inp1_ub <= 0)
        inp1_unsettled = (~inp1_positive & ~inp1_negative)
        inp2_positive = ((inp2_lb < 0) & (inp2_ub > 0))
        inp2_negative = (inp2_ub <= 0)
        inp2_unsettled = (~inp2_positive & ~inp2_negative)
        delta_positive = ((d_lb < 0) & (d_ub > 0))
        delta_negative = (d_ub <= 0)
        delta_unsettled = (~delta_positive & ~delta_negative)

        # --- update lb and ub of pre-activation ---
        pre_relu1.lb = inp1_lb
        pre_relu1.ub = inp1_ub
        pre_relu2.lb = inp2_lb
        pre_relu2.ub = inp2_ub
        pre_delta.lb = d_lb
        pre_delta.ub = d_ub

        # --- ReLU approximation for z1 ---
        if inp1_positive:
            model.addConstr(pre_relu1 >= 0, name=f"const_pre_relu1_active_{relu_layer_idx}_{pos}")
            model.addConstr(relu1 == pre_relu1, name=f"const_relu1_active_{relu_layer_idx}_{pos}")
        elif inp1_negative:
            model.addConstr(pre_relu1 <= 0, name=f"const_pre_relu1_inactive_{relu_layer_idx}_{pos}")
            model.addConstr(relu1 == 0, name=f"const_relu1_inactive_{relu_layer_idx}_{pos}")
        else:
            model.addConstr(relu1 >= 0, name=f"const_relu1_lb0_{relu_layer_idx}_{pos}")
            model.addConstr(relu1 >= pre_relu1, name=f"const_relu1_lb1_{relu_layer_idx}_{pos}")
            slope = inp1_ub / (inp1_ub - inp1_lb + 1e-6)
            intercept = -inp1_lb * slope
            model.addConstr(relu1 <= slope * pre_relu1 + intercept, name=f"const_relu1_ub_{relu_layer_idx}_{pos}")

        # --- ReLU approximation for z2 ---
        if inp2_positive:
            model.addConstr(pre_relu2 >= 0, name=f"const_pre_relu2_active_{relu_layer_idx}_{pos}")
            model.addConstr(relu2 == pre_relu2, name=f"const_relu2_active_{relu_layer_idx}_{pos}")
        elif inp2_negative:
            model.addConstr(pre_relu2 <= 0, name=f"const_pre_relu2_inactive_{relu_layer_idx}_{pos}")
            model.addConstr(relu2 == 0, name=f"const_relu2_inactive_{relu_layer_idx}_{pos}")
        else:
            model.addConstr(relu2 >= 0, name=f"const_relu2_lb0_{relu_layer_idx}_{pos}")
            model.addConstr(relu2 >= pre_relu2, name=f"const_relu2_lb1_{relu_layer_idx}_{pos}")
            slope = inp2_ub / (inp2_ub - inp2_lb + 1e-6)
            intercept = -inp2_lb * slope
            model.addConstr(relu2 <= slope * pre_relu2 + intercept, name=f"const_relu2_ub_{relu_layer_idx}_{pos}")

        # --- Delta for ReLU outputs ---
        # 1.
        if inp1_unsettled & inp2_negative & delta_positive:
            model.addConstr(delta <= pre_delta, name=f"const_delta_{relu_layer_idx}_{pos}")
        # 2.
        elif inp1_unsettled & inp2_positive:
            model.addConstr(delta >= pre_delta, name=f"const_delta_{relu_layer_idx}_{pos}")
        # 3.
        elif inp1_negative & inp2_unsettled & delta_negative:
            model.addConstr(delta >= pre_delta, name=f"const_delta_{relu_layer_idx}_{pos}")
        # 4.
        elif inp1_positive & inp2_unsettled:
            model.addConstr(delta <= pre_delta, name=f"const_delta_{relu_layer_idx}_{pos}")
        # 5.
        elif inp1_unsettled & inp2_unsettled & delta_positive:
            model.addConstr(delta <= pre_delta, name=f"const_delta_ub_{relu_layer_idx}_{pos}")
            model.addConstr(delta >= 0, name=f"const_delta_lb_{relu_layer_idx}_{pos}")
        # 6.
        elif inp1_unsettled & inp2_unsettled & delta_negative:
            model.addConstr(delta >= pre_delta, name=f"const_delta_lb_{relu_layer_idx}_{pos}")
            model.addConstr(delta <= 0, name=f"const_delta_ub_{relu_layer_idx}_{pos}")
        # 7.
        elif inp1_unsettled & inp2_unsettled & delta_unsettled:
            slope_ub = d_ub / (d_ub - d_lb + 1e-6)
            intercept_ub = -d_lb * slope_ub
            slope_lb = -d_lb / (d_ub - d_lb + 1e-6)
            intercept_lb = -d_ub * slope_lb

            model.addConstr(delta <= slope_ub * pre_delta + intercept_ub, name=f"const_delta_ub_{relu_layer_idx}_{pos}")
            model.addConstr(delta >= slope_lb * pre_delta + intercept_lb, name=f"const_delta_lb_{relu_layer_idx}_{pos}")

        model.update()
        return

    def add_ds_1_grb_constraints(self, model, DS_info, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        [d_lb, d_ub] -> [d_lb, split_value]
        """
        split_layer_idx = DS_info.layer_idx
        split_pos = DS_info.pos
        split_value = DS_info.split_value
        inp1_lb = inp1_lbs[split_layer_idx][split_pos].item()
        inp1_ub = inp1_ubs[split_layer_idx][split_pos].item()
        inp2_lb = inp2_lbs[split_layer_idx][split_pos].item()
        inp2_ub = inp2_ubs[split_layer_idx][split_pos].item()
        d_lb = d_lbs[split_layer_idx][split_pos].item()
        d_ub = d_ubs[split_layer_idx][split_pos].item()

        # remove old constraints
        self.remove_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=inp1_lb, inp1_ub=inp1_ub, inp2_lb=inp2_lb, inp2_ub=inp2_ub,
                                      d_lb=d_lb, d_ub=d_ub)

        # update bounds
        new_d_lb = d_lb
        new_d_ub = split_value
        new_inp1_lb = inp1_lb  # lb1 = lb2 + d_lb
        new_inp2_ub = inp2_ub  # ub2 = ub1 - d_lb
        new_inp1_ub = min(inp1_ub, inp2_ub + new_d_ub)  # ub1 = ub2 + new_d_ub
        new_inp2_lb = max(inp2_lb, inp1_lb - new_d_ub)  # lb2 = lb1 - new_d_ub

        # add new constraints
        self.update_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=new_inp1_lb, inp1_ub=new_inp1_ub, inp2_lb=new_inp2_lb, inp2_ub=new_inp2_ub,
                                      d_lb=new_d_lb, d_ub=new_d_ub)

        return

    def add_ds_2_grb_constraints(self, model, DS_info, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        [d_lb, d_ub] -> [split_value, d_ub]
        """
        split_layer_idx = DS_info.layer_idx
        split_pos = DS_info.pos
        split_value = DS_info.split_value
        inp1_lb = inp1_lbs[split_layer_idx][split_pos].item()
        inp1_ub = inp1_ubs[split_layer_idx][split_pos].item()
        inp2_lb = inp2_lbs[split_layer_idx][split_pos].item()
        inp2_ub = inp2_ubs[split_layer_idx][split_pos].item()
        d_lb = d_lbs[split_layer_idx][split_pos].item()
        d_ub = d_ubs[split_layer_idx][split_pos].item()

        # remove old constraints
        self.remove_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=inp1_lb, inp1_ub=inp1_ub, inp2_lb=inp2_lb, inp2_ub=inp2_ub,
                                      d_lb=d_lb, d_ub=d_ub)

        # update bounds
        new_d_lb = split_value
        new_d_ub = d_ub
        new_inp1_ub = inp1_ub  # ub1 = ub2 + d_ub
        new_inp2_lb = inp2_lb  # lb2 = lb1 - d_ub
        new_inp1_lb = max(inp1_lb, inp2_lb + new_d_lb)  # lb1 = lb2 + new_d_lb
        new_inp2_ub = min(inp2_ub, inp1_ub - new_d_lb)  # ub2 = ub1 - new_d_lb

        # add new constraints
        self.update_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=new_inp1_lb, inp1_ub=new_inp1_ub, inp2_lb=new_inp2_lb, inp2_ub=new_inp2_ub,
                                      d_lb=new_d_lb, d_ub=new_d_ub)
        return

    def add_ns_A1_grb_constraints(self, model, NS_info, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        [l1, u1] -> [l1, 0]
        """
        split_layer_idx = NS_info.layer_idx
        split_pos = NS_info.pos
        split_value = NS_info.split_value
        inp1_lb = inp1_lbs[split_layer_idx][split_pos].item()
        inp1_ub = inp1_ubs[split_layer_idx][split_pos].item()
        inp2_lb = inp2_lbs[split_layer_idx][split_pos].item()
        inp2_ub = inp2_ubs[split_layer_idx][split_pos].item()
        d_lb = d_lbs[split_layer_idx][split_pos].item()
        d_ub = d_ubs[split_layer_idx][split_pos].item()

        # remove old constraints
        self.remove_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=inp1_lb, inp1_ub=inp1_ub, inp2_lb=inp2_lb, inp2_ub=inp2_ub,
                                      d_lb=d_lb, d_ub=d_ub)

        # update bounds
        new_inp1_lb = inp1_lb  # lb1 = lb2 + d_lb
        new_inp2_lb = inp2_lb  # lb2 = lb1 - d_ub
        new_d_lb = d_lb  # d_lb = lb1 - ub2
        new_inp1_ub = split_value  # ub1 = 0
        new_inp2_ub = min(inp2_ub, -d_lb)  # ub2 = ub1 - d_lb
        new_d_ub = min(d_ub, -inp2_lb)  # d_ub = ub1 - lb2

        # add new constraints
        self.update_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=new_inp1_lb, inp1_ub=new_inp1_ub, inp2_lb=new_inp2_lb, inp2_ub=new_inp2_ub,
                                      d_lb=new_d_lb, d_ub=new_d_ub)
        return

    def add_ns_A2_grb_constraints(self, model, NS_info, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        [l1, u1] -> [0, u1]
        """
        split_layer_idx = NS_info.layer_idx
        split_pos = NS_info.pos
        split_value = NS_info.split_value
        inp1_lb = inp1_lbs[split_layer_idx][split_pos].item()
        inp1_ub = inp1_ubs[split_layer_idx][split_pos].item()
        inp2_lb = inp2_lbs[split_layer_idx][split_pos].item()
        inp2_ub = inp2_ubs[split_layer_idx][split_pos].item()
        d_lb = d_lbs[split_layer_idx][split_pos].item()
        d_ub = d_ubs[split_layer_idx][split_pos].item()

        # remove old constraints
        self.remove_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=inp1_lb, inp1_ub=inp1_ub, inp2_lb=inp2_lb, inp2_ub=inp2_ub,
                                      d_lb=d_lb, d_ub=d_ub)

        # update bounds
        new_inp1_ub = inp1_ub  # ub1 = ub2 + d_ub
        new_inp2_ub = inp2_ub  # ub2 = ub1 - d_lb
        new_d_ub = d_ub  # d_ub = ub1 - lb2
        new_inp1_lb = split_value  # lb1 = 0
        new_inp2_lb = max(inp2_lb, -d_ub)  # lb2 = lb1 - d_ub
        new_d_lb = min(d_lb, -inp2_ub)  # d_lb = lb1 - ub2

        # add new constraints
        self.update_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=new_inp1_lb, inp1_ub=new_inp1_ub, inp2_lb=new_inp2_lb, inp2_ub=new_inp2_ub,
                                      d_lb=new_d_lb, d_ub=new_d_ub)
        return

    def add_ns_B1_grb_constraints(self, model, NS_info, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        [l2, u2] -> [l2, 0]
        """
        split_layer_idx = NS_info.layer_idx
        split_pos = NS_info.pos
        split_value = NS_info.split_value
        inp1_lb = inp1_lbs[split_layer_idx][split_pos].item()
        inp1_ub = inp1_ubs[split_layer_idx][split_pos].item()
        inp2_lb = inp2_lbs[split_layer_idx][split_pos].item()
        inp2_ub = inp2_ubs[split_layer_idx][split_pos].item()
        d_lb = d_lbs[split_layer_idx][split_pos].item()
        d_ub = d_ubs[split_layer_idx][split_pos].item()

        # remove old constraints
        self.remove_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=inp1_lb, inp1_ub=inp1_ub, inp2_lb=inp2_lb, inp2_ub=inp2_ub,
                                      d_lb=d_lb, d_ub=d_ub)

        # update bounds
        new_inp1_lb = inp1_lb  # lb1 = lb2 + d_lb
        new_inp2_lb = inp2_lb  # lb2 = lb1 - d_ub
        new_d_ub = d_ub  # d_ub = ub1 - lb2
        new_inp2_ub = split_value  # ub2 = 0
        new_inp1_ub = min(inp1_ub, d_ub)  # ub1 = ub2 + d_ub
        new_d_lb = max(d_lb, inp1_lb)  # d_lb = lb1 - ub2

        # add new constraints
        self.update_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=new_inp1_lb, inp1_ub=new_inp1_ub, inp2_lb=new_inp2_lb, inp2_ub=new_inp2_ub,
                                      d_lb=new_d_lb, d_ub=new_d_ub)
        return

    def add_ns_B2_grb_constraints(self, model, NS_info, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        [l2, u2] -> [0, u2]
        """
        split_layer_idx = NS_info.layer_idx
        split_pos = NS_info.pos
        split_value = NS_info.split_value
        inp1_lb = inp1_lbs[split_layer_idx][split_pos].item()
        inp1_ub = inp1_ubs[split_layer_idx][split_pos].item()
        inp2_lb = inp2_lbs[split_layer_idx][split_pos].item()
        inp2_ub = inp2_ubs[split_layer_idx][split_pos].item()
        d_lb = d_lbs[split_layer_idx][split_pos].item()
        d_ub = d_ubs[split_layer_idx][split_pos].item()

        # remove old constraints
        self.remove_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=inp1_lb, inp1_ub=inp1_ub, inp2_lb=inp2_lb, inp2_ub=inp2_ub,
                                      d_lb=d_lb, d_ub=d_ub)

        # update bounds
        new_inp1_ub = inp1_ub  # ub1 = ub2 + d_ub
        new_inp2_ub = inp2_ub  # ub2 = ub1 - d_lb
        new_d_lb = d_lb  # d_lb = lb1 - ub2
        new_inp2_lb = split_value  # lb2 = 0
        new_inp1_lb = max(inp1_lb, d_lb)  # lb1 = lb2 + d_lb
        new_d_ub = min(d_ub, inp1_ub)  # d_ub = ub1 - lb2

        # add new constraints
        self.update_model_constraints(model=model, layer_idx=split_layer_idx, pos=split_pos,
                                      inp1_lb=new_inp1_lb, inp1_ub=new_inp1_ub, inp2_lb=new_inp2_lb, inp2_ub=new_inp2_ub,
                                      d_lb=new_d_lb, d_ub=new_d_ub)
        return


    def get_out_diff(self, IARb):
        inp1_vars = self.grb_inp1_vars[0].tolist()
        inp2_vars = self.grb_inp2_vars[0].tolist()

        input_shape = tuple(IARb.input_shape)
        inp1_values = [v.X for v in inp1_vars]
        input1_tensor = torch.tensor(inp1_values, dtype=torch.float32).reshape(1, *input_shape)
        output1 = self.run_net_forward(IARb, input1_tensor)
        inp2_values = [v.X for v in inp2_vars]
        input2_tensor = torch.tensor(inp2_values, dtype=torch.float32).reshape(1, *input_shape)
        output2 = self.run_net_forward(IARb, input2_tensor)

        correct_label_idx = self.inp1_correct_label
        diff = output1[0, correct_label_idx] - output2[0, correct_label_idx]
        return diff.item()

    def run_net_forward(self, IARb, input_tensor):
        x = input_tensor

        for layer in IARb.net:
            if layer.type == LayerType.Conv2D:
                # conv2d expects shape [N, C_in, H_in, W_in]
                x = F.conv2d(
                    x, layer.weight, layer.bias,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation
                )

            elif layer.type == LayerType.ReLU:
                x = F.relu(x)

            elif layer.type == LayerType.Linear:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)  # flatten
                x = F.linear(x, layer.weight, layer.bias)

            else:
                raise NotImplementedError(f"Unsupported layer type: {layer.type}")

        return x

    def run_label_diff_lp_analysis(self, model, delta_out_grb_vars, IARb, log_file=""):
        """
        Run LP analysis for output label differences between each input pair.
        """
        delta_output_bounds = {}
        if self.global_target:
            correct_label_idx = self.inp1_correct_label
            target = delta_out_grb_vars[correct_label_idx]

            delta_min = self.minimize_lp(model, target)
            if delta_min is None:
                self.status = Status.UNREACHABLE
                with open(f"{log_file}log.md", 'a') as f:
                    f.write(f"Optimization infeasible because this subproblem isn't reachable.\n")
                return self.status, None
            else:
                diff = self.get_out_diff(IARb)
                if self.threshold < abs(diff):
                    return Status.ADV_EXAMPLE, None

            delta_max = self.maximize_lp(model, target)
            if delta_max is None:
                self.status = Status.UNREACHABLE
                with open(f"{log_file}log.md", 'a') as f:
                    f.write(f"Optimization infeasible because this subproblem isn't reachable.\n")
                return self.status, None
            else:
                diff = self.get_out_diff(IARb)
                if self.threshold < abs(diff):
                    return Status.ADV_EXAMPLE, None

            delta_output_bounds[correct_label_idx] = [delta_min, delta_max]

            if self.threshold is not None:
                if max(abs(delta_min), abs(delta_max)) <= self.threshold:
                    self.status = Status.VERIFIED
                else:
                    self.status = Status.UNKNOWN
        else:
            for delta_idx in range(delta_out_grb_vars.shape[0]):
                target = delta_out_grb_vars[delta_idx]
                delta_min = self.minimize_lp(model, target)
                if delta_min is None:
                    self.status = Status.UNREACHABLE
                    with open(f"{log_file}log.md", 'a') as f:
                        f.write(f"Optimization infeasible because this subproblem isn't reachable.\n")
                    return self.status, None
                delta_max = self.maximize_lp(model, target)
                if delta_max is None:
                    self.status = Status.UNREACHABLE
                    with open(f"{log_file}log.md", 'a') as f:
                        f.write(f"Optimization infeasible because this subproblem isn't reachable.\n")
                    return self.status, None
                delta_output_bounds[delta_idx] = [delta_min, delta_max]
            # find the maximum absolute bound among all dimensions
            index_max, abs_max_bound = None, None
            for i, bounds in delta_output_bounds.items():
                curr_abs_bound = max(abs(bounds[0]), abs(bounds[1]))
                if abs_max_bound is None or curr_abs_bound > abs_max_bound:
                    abs_max_bound = curr_abs_bound
                    index_max = i
        return self.status, delta_output_bounds

    def run_global_robustness_analysis(self, IARb, DS_history=None, NS_history=None, log_file=""):
        if self.lp_analysis:
            inp1_lbs = IARb.inp1_lbs
            inp1_ubs = IARb.inp1_ubs
            inp2_lbs = IARb.inp2_lbs
            inp2_ubs = IARb.inp2_ubs
            d_lbs = IARb.d_lbs
            d_ubs = IARb.d_ubs

            if len(inp1_lbs[-1]) != len(inp2_lbs[-1]):
                raise ValueError("The number of labels for inp1 and inp2 must be the same.")

            # $ LP for each label difference
            if self.grb_model is None:
                self.grb_model = self.create_lp_model(net=IARb.net, shapes=IARb.shapes,
                                                      inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs, inp2_lbs=inp2_lbs,
                                                      inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)
                temp_relAna = RelationalAnalysis()
                temp_relAna.duplicate(self)
            else:
                temp_relAna = RelationalAnalysis()
                temp_relAna.duplicate(self)
                temp_relAna.grb_model = temp_relAna.update_lp_model(model=temp_relAna.grb_model, DS_history=DS_history, NS_history=NS_history,
                                                                    inp1_lbs=inp1_lbs, inp1_ubs=inp1_ubs, inp2_lbs=inp2_lbs,
                                                                    inp2_ubs=inp2_ubs, d_lbs=d_lbs, d_ubs=d_ubs)

            curr_status, delta_output = \
                temp_relAna.run_label_diff_lp_analysis(model=temp_relAna.grb_model, delta_out_grb_vars=temp_relAna.grb_delta_vars[-1], IARb=IARb, log_file=log_file)

            # trash temp_relAna
            self.trash_used_relAna(temp_relAna=temp_relAna)

            return curr_status, delta_output

        else:
            raise NotImplementedError("Non-LP global robustness analysis is not implemented yet.")
