
import torch
import numpy as np
from relu.relu import ReluInput, ReluApprox, ReluCondition


class ReLUTransformer:
    def get_relu_input_condition(self, relu_input_layer_idx,
                                 inp1_lb_layer, inp1_ub_layer,
                                 inp2_lb_layer, inp2_ub_layer):

        inp1_positive = (inp1_lb_layer >= 0) & (inp1_ub_layer > 0)
        inp1_negative = (inp1_ub_layer <= 0)
        inp1_unstable_y0 = ~(inp1_positive) & ~(inp1_negative) & (inp1_ub_layer < -inp1_lb_layer)
        inp1_unstable_yx = ~(inp1_positive) & ~(inp1_negative) & (inp1_ub_layer >= -inp1_lb_layer)

        inp2_positive = (inp2_lb_layer >= 0) & (inp2_ub_layer > 0)
        inp2_negative = (inp2_ub_layer <= 0)
        inp2_unstable_y0 = ~(inp2_positive) & ~(inp2_negative) & (inp2_ub_layer < -inp2_lb_layer)
        inp2_unstable_yx = ~(inp2_positive) & ~(inp2_negative) & (inp2_ub_layer >= -inp2_lb_layer)

        inp1_relu_condition = torch.full_like(inp1_lb_layer, fill_value=ReluCondition.UNINITIALIZED.value, dtype=torch.int32)
        inp1_relu_condition[inp1_positive] = ReluCondition.POSITIVE.value
        inp1_relu_condition[inp1_negative] = ReluCondition.NEGATIVE.value
        inp1_relu_condition[inp1_unstable_y0] = ReluCondition.UNSTABLE_0.value
        inp1_relu_condition[inp1_unstable_yx] = ReluCondition.UNSTABLE_1.value
        if (inp1_relu_condition == ReluCondition.UNINITIALIZED.value).any():
            raise ValueError("Input 1 relu condition is not initialized properly.")

        inp2_relu_condition = torch.full_like(inp2_lb_layer, fill_value=ReluCondition.UNINITIALIZED.value, dtype=torch.int32)
        inp2_relu_condition[inp2_positive] = ReluCondition.POSITIVE.value
        inp2_relu_condition[inp2_negative] = ReluCondition.NEGATIVE.value
        inp2_relu_condition[inp2_unstable_y0] = ReluCondition.UNSTABLE_0.value
        inp2_relu_condition[inp2_unstable_yx] = ReluCondition.UNSTABLE_1.value
        if (inp2_relu_condition == ReluCondition.UNINITIALIZED.value).any():
            raise ValueError("Input 2 relu condition is not initialized properly.")

        if self.inp1_relu_input_info[relu_input_layer_idx] is None:
            self.inp1_relu_input_info[relu_input_layer_idx] = inp1_relu_condition
            self.inp2_relu_input_info[relu_input_layer_idx] = inp2_relu_condition
        else:
            inp1_stable = (inp1_relu_condition == ReluCondition.POSITIVE.value) | (inp1_relu_condition == ReluCondition.NEGATIVE.value)
            inp2_stable = (inp2_relu_condition == ReluCondition.POSITIVE.value) | (inp2_relu_condition == ReluCondition.NEGATIVE.value)
            self.inp1_relu_input_info[relu_input_layer_idx][inp1_stable] = inp1_relu_condition[inp1_stable]
            self.inp2_relu_input_info[relu_input_layer_idx][inp2_stable] = inp2_relu_condition[inp2_stable]

        return self.inp1_relu_input_info[relu_input_layer_idx], self.inp2_relu_input_info[relu_input_layer_idx]

    def handle_relu_normal(self, back_prop_struct, relu_input_layer_idx,
                           inp1_lb_layer, inp1_ub_layer,
                           inp2_lb_layer, inp2_ub_layer,
                           d_lb_layer, d_ub_layer):
        """
        relu transformation with normal propagation method.
        """
        # print("normal")

        # $ relu condition -->
        inp1_relu_condition, inp2_relu_condition = \
            self.get_relu_input_condition(relu_input_layer_idx,
                                          inp1_lb_layer, inp1_ub_layer,
                                          inp2_lb_layer, inp2_ub_layer)

        inp1_positive = (inp1_relu_condition == ReluCondition.POSITIVE.value)
        inp1_negative = (inp1_relu_condition == ReluCondition.NEGATIVE.value)
        inp1_unstable_0 = (inp1_relu_condition == ReluCondition.UNSTABLE_0.value)  # (inp1_ub_layer < -inp1_lb_layer)
        inp1_unstable_1 = (inp1_relu_condition == ReluCondition.UNSTABLE_1.value)  # otherwise
        inp1_unstable = inp1_unstable_0 | inp1_unstable_1
        inp2_positive = (inp2_relu_condition == ReluCondition.POSITIVE.value)
        inp2_negative = (inp2_relu_condition == ReluCondition.NEGATIVE.value)
        inp2_unstable_0 = (inp2_relu_condition == ReluCondition.UNSTABLE_0.value)  # (inp2_ub_layer < -inp2_lb_layer)
        inp2_unstable_1 = (inp2_relu_condition == ReluCondition.UNSTABLE_1.value)  # otherwise
        inp2_unstable = inp2_unstable_0 | inp2_unstable_1

        delta_positive = (d_lb_layer >= 0) & (d_ub_layer > 0)
        delta_negative = (d_ub_layer <= 0)
        delta_unstable = ~(delta_positive) & ~(delta_negative)
        # $ <-- relu condition

        # $ initialize -->
        lambda_lb = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_lb_inp1 = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp1 = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_lb_inp2 = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp2 = torch.zeros(inp1_lb_layer.size(), device=self.device)

        lambda_lb_inp1_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp1_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        mu_ub_inp1_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_lb_inp2_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp2_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        mu_ub_inp2_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)

        mu_lb = torch.zeros(inp1_lb_layer.size(), device=self.device)
        mu_ub = torch.zeros(inp1_lb_layer.size(), device=self.device)
        # $ <-- initialize

        # $ for individual bounds -->
        """
        Y = CX + b --> Y = C(LX' + m) + b = CLX' + b +Cm
        
        * y = ReLU(x) *
        (positive) x <= y <= x
        (negative) 0 <= y <= 0 #note: initially y = 0
        (unstable) (0 or x) <= y <= u(x-l)/(u-l) --> y = lx + m
        """

        # $ inp1 is positive
        lambda_lb_inp1_prop = torch.where(inp1_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1_prop)
        lambda_ub_inp1_prop = torch.where(inp1_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp1_prop)
        # $ inp1 is unstable
        # * lb = 0 nothing to do
        # * lb = x
        # lambda_lb_inp1_prop = torch.where(inp1_unstable, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1_prop)
        # * lb = 0 (u ≤ -l) or x (u > -l)
        temp_inp1_pm = torch.where(inp1_unstable_0, torch.zeros(inp1_lb_layer.size(), device=self.device), torch.ones(inp1_lb_layer.size(), device=self.device))
        lambda_lb_inp1_prop = torch.where(inp1_unstable, temp_inp1_pm, lambda_lb_inp1_prop)

        lambda_ub_inp1_prop = torch.where(inp1_unstable, inp1_ub_layer/(inp1_ub_layer - inp1_lb_layer + 1e-15), lambda_ub_inp1_prop)
        mu_ub_inp1_prop = torch.where(inp1_unstable, -(inp1_ub_layer * inp1_lb_layer) / (inp1_ub_layer - inp1_lb_layer + 1e-15), mu_ub_inp1_prop)

        # $ inp2 is positive
        lambda_lb_inp2_prop = torch.where(inp2_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp2_prop)
        lambda_ub_inp2_prop = torch.where(inp2_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp2_prop)
        # $ inp2 is unstable
        # * lb = 0 nothing to do
        # * lb = x
        # lambda_lb_inp2_prop = torch.where(inp2_unstable, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp2_prop)
        # * lb = 0 (u ≤ -l) or x (u > -l)
        temp_inp2_pm = torch.where(inp2_unstable_0, torch.zeros(inp1_lb_layer.size(), device=self.device), torch.ones(inp1_lb_layer.size(), device=self.device))
        lambda_lb_inp2_prop = torch.where(inp2_unstable, temp_inp2_pm, lambda_lb_inp2_prop)

        lambda_ub_inp2_prop = torch.where(inp2_unstable, inp2_ub_layer/(inp2_ub_layer - inp2_lb_layer + 1e-15), lambda_ub_inp2_prop)
        mu_ub_inp2_prop = torch.where(inp2_unstable, -(inp2_ub_layer * inp2_lb_layer) / (inp2_ub_layer - inp2_lb_layer + 1e-15), mu_ub_inp2_prop)
        # $ <-- for individual bounds

        """
        (case 1) x1.ub <= 0 and x2.ub <= 0
        Dy = 0
		#note: initially Dy = 0
		"""
        """
        (case 2) x1.lb >=  0 and x2.lb >= 0
        Dy = Dx
        """
        case_2 = inp1_positive & inp2_positive
        lambda_lb = torch.where(case_2, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(case_2, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub)
        """
        (case 3) x1.lb >= 0 and x2.ub <= 0
		Dy = x1
        """
        case_3 = inp1_positive & inp2_negative
        lambda_lb_inp1 = torch.where(case_3, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1)
        lambda_ub_inp1 = torch.where(case_3, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp1)
        """
        (case 4) x1.ub <= 0 and x2.lb >= 0
		Dy = -x2
        """
        case_4 = inp1_negative & inp2_positive
        lambda_lb_inp2 = torch.where(case_4, -torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp2)
        lambda_ub_inp2 = torch.where(case_4, -torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp2)
        """
        (case 5) (x1.lb < 0 and x1.ub > 0) and x2.ub <= 0
		Dy = y1
		Dy.lb = lx1.lb
		Dy.ub = lx1.ub + m.ub
        """
        case_5 = inp1_unstable & inp2_negative
        lambda_lb_inp1 = torch.where(case_5, lambda_lb_inp1_prop, lambda_lb_inp1)
        lambda_ub_inp1 = torch.where(case_5, lambda_ub_inp1_prop, lambda_ub_inp1)
        mu_ub = torch.where(case_5, mu_ub_inp1_prop, mu_ub)
        """
        (case 6) x1.ub <= 0 and (x2.lb < 0 and x2.ub > 0)
		Dy = -y2
		Dy.lb = -lx2.ub - m.ub
		Dy.ub = -lx2.lb
        """
        case_6 = inp1_negative & inp2_unstable
        lambda_lb_inp2 = torch.where(case_6, -lambda_ub_inp2_prop, lambda_lb_inp2)
        mu_lb = torch.where(case_6, -mu_ub_inp2_prop, mu_lb)
        lambda_ub_inp2 = torch.where(case_6, -lambda_lb_inp2_prop, lambda_ub_inp2)
        """
        (case 7) (x1.lb < 0 and x1.ub > 0) and x2.lb >= 0
		Dy = y1 - x2
		Dy.lb = Dx
		Dy.ub = lx1.ub + m.ub - x2
        """
        case_7 = inp1_unstable & inp2_positive
        lambda_lb = torch.where(case_7, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb)
        lambda_ub_inp1 = torch.where(case_7, lambda_ub_inp1_prop, lambda_ub_inp1)
        mu_ub = torch.where(case_7, mu_ub_inp1_prop, mu_ub)
        lambda_ub_inp2 = torch.where(case_7, -torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp2)
        """
        (case 8) x1.lb >= 0 and (x2.lb < 0 and x2.ub > 0)
		Dy = x1 - y2
		Dy.lb = x1 - lx2.ub - m.ub
		Dy.ub = Dx
        """
        case_8 = inp1_positive & inp2_unstable
        lambda_lb_inp1 = torch.where(case_8, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1)
        lambda_lb_inp2 = torch.where(case_8, -lambda_ub_inp2_prop, lambda_lb_inp2)
        mu_lb = torch.where(case_8, -mu_ub_inp2_prop, mu_lb)
        lambda_ub = torch.where(case_8, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub)
        """
        (case 9) (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb >= 0)
        Dy.lb = 0
        Dy.ub = y1 - y2 or Dx --> Dx
        note: "refine bounds" produce Dx.ub ≤ x1.ub - x2.lb
        """
        case_9 = inp1_unstable & inp2_unstable & delta_positive
        lambda_ub = torch.where(case_9, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub)
        """
        (case 10) (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb <= 0)
        Dy.lb = Dx or y1 - y2 --> Dx 
        note: "refine bounds" produce Dx.lb ≥ x1.lb - x2.ub
        Dy.ub = 0
        """
        case_10 = inp1_unstable & inp2_unstable & delta_negative
        lambda_lb = torch.where(case_10, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb)

        """
        (case 11) (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb < 0 and delta_ub > 0)
        """
        temp_mu = (d_lb_layer * d_ub_layer) / (d_ub_layer - d_lb_layer + 1e-15)
        temp_lambda_lb = (-d_lb_layer) / (d_ub_layer - d_lb_layer + 1e-15)
        temp_lambda_ub = d_ub_layer / (d_ub_layer - d_lb_layer + 1e-15)

        case_11 = inp1_unstable & inp2_unstable & delta_unstable
        lambda_lb = torch.where(case_11, temp_lambda_lb, lambda_lb)
        lambda_ub = torch.where(case_11, temp_lambda_ub, lambda_ub)
        mu_lb = torch.where(case_11, temp_mu, mu_lb)
        mu_ub = torch.where(case_11, -temp_mu, mu_ub)

        # Segregate the +ve and -ve components of the coefficients
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.d_C_lb)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.d_C_ub)
        neg_comp_lb_inp1, pos_comp_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_lb)
        neg_comp_ub_inp1, pos_comp_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_ub)
        neg_comp_lb_inp2, pos_comp_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_lb)
        neg_comp_ub_inp2, pos_comp_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_ub)

        neg_coef_lb_inp1, pos_coef_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.x1_C_lb)
        neg_coef_ub_inp1, pos_coef_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.x1_C_ub)
        neg_coef_lb_inp2, pos_coef_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.x2_C_lb)
        neg_coef_ub_inp2, pos_coef_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.x2_C_ub)

        d_C_lb = pos_comp_lb * lambda_lb + neg_comp_lb * lambda_ub
        d_x1_C_lb = pos_comp_lb_inp1 * lambda_lb_inp1_prop + neg_comp_lb_inp1 * lambda_ub_inp1_prop
        d_x1_C_lb = d_x1_C_lb + pos_comp_lb * lambda_lb_inp1 + neg_comp_lb * lambda_ub_inp1
        d_x2_C_lb = pos_comp_lb_inp2 * lambda_lb_inp2_prop + neg_comp_lb_inp2 * lambda_ub_inp2_prop
        d_x2_C_lb = d_x2_C_lb + pos_comp_lb * lambda_lb_inp2 + neg_comp_lb * lambda_ub_inp2

        d_C_ub = pos_comp_ub * lambda_ub + neg_comp_ub * lambda_lb
        d_x1_C_ub = pos_comp_ub_inp1 * lambda_ub_inp1_prop + neg_comp_ub_inp1 * lambda_lb_inp1_prop
        d_x1_C_ub = d_x1_C_ub + pos_comp_ub * lambda_ub_inp1 + neg_comp_ub * lambda_lb_inp1
        d_x2_C_ub = pos_comp_ub_inp2 * lambda_ub_inp2_prop + neg_comp_ub_inp2 * lambda_lb_inp2_prop
        d_x2_C_ub = d_x2_C_ub + pos_comp_ub * lambda_ub_inp2 + neg_comp_ub * lambda_lb_inp2

        d_b_lb = pos_comp_lb @ mu_lb + neg_comp_lb @ mu_ub + back_prop_struct.d_b_lb
        d_b_lb = d_b_lb + neg_comp_lb_inp1 @ mu_ub_inp1_prop + neg_comp_lb_inp2 @ mu_ub_inp2_prop
        d_b_ub = pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb + back_prop_struct.d_b_ub
        d_b_ub = d_b_ub + pos_comp_ub_inp1 @ mu_ub_inp1_prop + pos_comp_ub_inp2 @ mu_ub_inp2_prop

        if back_prop_struct.x1_b_lb is not None:
            x1_b_lb = back_prop_struct.x1_b_lb + neg_coef_lb_inp1 @ mu_ub_inp1_prop
            x1_b_ub = back_prop_struct.x1_b_ub + pos_coef_ub_inp1 @ mu_ub_inp1_prop
            x2_b_lb = back_prop_struct.x2_b_lb + neg_coef_lb_inp2 @ mu_ub_inp2_prop
            x2_b_ub = back_prop_struct.x2_b_ub + pos_coef_ub_inp2 @ mu_ub_inp2_prop

            x1_C_lb = neg_coef_lb_inp1 * lambda_ub_inp1_prop + pos_coef_lb_inp1 * lambda_lb_inp1_prop
            x1_C_ub = neg_coef_ub_inp1 * lambda_lb_inp1_prop + pos_coef_ub_inp1 * lambda_ub_inp1_prop
            x2_C_lb = neg_coef_lb_inp2 * lambda_ub_inp2_prop + pos_coef_lb_inp2 * lambda_lb_inp2_prop
            x2_C_ub = neg_coef_ub_inp2 * lambda_lb_inp2_prop + pos_coef_ub_inp2 * lambda_ub_inp2_prop
        else:
            x1_b_lb = None
            x1_b_ub = None
            x2_b_lb = None
            x2_b_ub = None

            x1_C_lb = None
            x1_C_ub = None
            x2_C_lb = None
            x2_C_ub = None

        back_prop_struct.populate(d_C_lb=d_C_lb, d_b_lb=d_b_lb,
                                  d_C_ub=d_C_ub, d_b_ub=d_b_ub,
                                  d_x1_C_lb=d_x1_C_lb,
                                  d_x1_C_ub=d_x1_C_ub,
                                  d_x2_C_lb=d_x2_C_lb,
                                  d_x2_C_ub=d_x2_C_ub,
                                  x1_C_lb=x1_C_lb, x2_C_lb=x2_C_lb,
                                  x1_C_ub=x1_C_ub, x2_C_ub=x2_C_ub,
                                  x1_b_lb=x1_b_lb, x2_b_lb=x2_b_lb,
                                  x1_b_ub=x1_b_ub, x2_b_ub=x2_b_ub)

        return back_prop_struct

    def handle_relu_DP(self, back_prop_struct, relu_input_layer_idx,
                       inp1_lb_layer, inp1_ub_layer,
                       inp2_lb_layer, inp2_ub_layer,
                       d_lb_layer, d_ub_layer):
        """
        relu transformation for two inputs with delta based on DiffPoly
        """
        # print("DP")

        # $ relu condition -->
        inp1_relu_condition, inp2_relu_condition = \
            self.get_relu_input_condition(relu_input_layer_idx,
                                          inp1_lb_layer, inp1_ub_layer,
                                          inp2_lb_layer, inp2_ub_layer)

        inp1_positive = (inp1_relu_condition == ReluCondition.POSITIVE.value)
        inp1_negative = (inp1_relu_condition == ReluCondition.NEGATIVE.value)
        inp1_unstable_0 = (inp1_relu_condition == ReluCondition.UNSTABLE_0.value)  # (inp1_ub_layer < -inp1_lb_layer)
        inp1_unstable_1 = (inp1_relu_condition == ReluCondition.UNSTABLE_1.value)  # otherwise
        inp1_unstable = inp1_unstable_0 | inp1_unstable_1
        inp2_positive = (inp2_relu_condition == ReluCondition.POSITIVE.value)
        inp2_negative = (inp2_relu_condition == ReluCondition.NEGATIVE.value)
        inp2_unstable_0 = (inp2_relu_condition == ReluCondition.UNSTABLE_0.value)  # (inp2_ub_layer < -inp2_lb_layer)
        inp2_unstable_1 = (inp2_relu_condition == ReluCondition.UNSTABLE_1.value)  # otherwise
        inp2_unstable = inp2_unstable_0 | inp2_unstable_1

        delta_positive = (d_lb_layer >= 0) & (d_ub_layer > 0)
        delta_negative = (d_ub_layer <= 0)
        delta_unstable = ~(delta_positive) & ~(delta_negative)
        # $ <-- relu condition

        # $ initialize -->
        lambda_lb = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_lb_inp1 = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp1 = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_lb_inp2 = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp2 = torch.zeros(inp1_lb_layer.size(), device=self.device)

        lambda_lb_inp1_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp1_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        mu_ub_inp1_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_lb_inp2_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        lambda_ub_inp2_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)
        mu_ub_inp2_prop = torch.zeros(inp1_lb_layer.size(), device=self.device)

        mu_lb = torch.zeros(inp1_lb_layer.size(), device=self.device)
        mu_ub = torch.zeros(inp1_lb_layer.size(), device=self.device)
        # $ <-- initialize

        # $ for individual bounds -->
        """
        Y = CX + b --> Y = C(LX' + m) + b = CLX' + b +Cm
        
        * y = ReLU(x) *
        (positive) x <= y <= x
        (negative) 0 <= y <= 0 #note: initially y = 0
        (unstable) (0 or x) <= y <= u(x-l)/(u-l) --> y = lx + m
        """

        # $ inp1 is positive
        lambda_lb_inp1_prop = torch.where(inp1_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1_prop)
        lambda_ub_inp1_prop = torch.where(inp1_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp1_prop)
        # $ inp1 is unstable
        # * limit output lower bound of relu to 0 -->
        temp_inp1_pm = torch.where(inp1_unstable_0, torch.zeros(inp1_lb_layer.size(), device=self.device), torch.ones(inp1_lb_layer.size(), device=self.device))
        lambda_lb_inp1_prop = torch.where(inp1_unstable, temp_inp1_pm, lambda_lb_inp1_prop)
        # * <-- limit
        lambda_ub_inp1_prop = torch.where(inp1_unstable, inp1_ub_layer/(inp1_ub_layer - inp1_lb_layer + 1e-15), lambda_ub_inp1_prop)
        mu_ub_inp1_prop = torch.where(inp1_unstable, -(inp1_ub_layer * inp1_lb_layer) / (inp1_ub_layer - inp1_lb_layer + 1e-15), mu_ub_inp1_prop)

        # $ inp2 is positive
        lambda_lb_inp2_prop = torch.where(inp2_positive, torch.ones(inp2_lb_layer.size(), device=self.device), lambda_lb_inp2_prop)
        lambda_ub_inp2_prop = torch.where(inp2_positive, torch.ones(inp2_lb_layer.size(), device=self.device), lambda_ub_inp2_prop)
        # $ inp2 is unstable
        # * limit output lower bound of relu to 0 -->
        temp_inp2_pm = torch.where(inp2_unstable_0, torch.zeros(inp2_lb_layer.size(), device=self.device), torch.ones(inp2_lb_layer.size(), device=self.device))
        lambda_lb_inp2_prop = torch.where(inp2_unstable, temp_inp2_pm, lambda_lb_inp2_prop)
        # * <-- limit
        lambda_ub_inp2_prop = torch.where(inp2_unstable, inp2_ub_layer/(inp2_ub_layer - inp2_lb_layer + 1e-15), lambda_ub_inp2_prop)
        mu_ub_inp2_prop = torch.where(inp2_unstable, -(inp2_ub_layer * inp2_lb_layer) / (inp2_ub_layer - inp2_lb_layer + 1e-15), mu_ub_inp2_prop)
        # $ <-- for individual bounds

        """
        (case 1) x1.ub <= 0 and x2.ub <= 0
        Dy = 0
		#note: initially Dy = 0
		"""
        """
        (case 2) x1.lb >=  0 and x2.lb >= 0
        Dy = Dx
        """
        lambda_lb = torch.where(inp1_positive & inp2_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(inp1_positive & inp2_positive, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub)
        """
        (case 3) x1.lb >= 0 and x2.ub <= 0
		Dy = x1
        """
        lambda_lb_inp1 = torch.where(inp1_positive & inp2_negative, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1)
        lambda_ub_inp1 = torch.where(inp1_positive & inp2_negative, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp1)
        """
        (case 4) x1.ub <= 0 and x2.lb >= 0
		Dy = -x2
        """
        lambda_lb_inp2 = torch.where(inp1_negative & inp2_positive, -torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp2)
        lambda_ub_inp2 = torch.where(inp1_negative & inp2_positive, -torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp2)
        """
        (case 5) (x1.lb < 0 and x1.ub > 0) and x2.ub <= 0
		Dy = y1
		Dy.lb = lx1.lb
		Dy.ub = lx1.ub + m.ub
        """
        case_5 = inp1_unstable & inp2_negative
        lambda_lb_inp1 = torch.where(case_5, lambda_lb_inp1_prop, lambda_lb_inp1)
        lambda_ub_inp1 = torch.where(case_5, lambda_ub_inp1_prop, lambda_ub_inp1)
        mu_ub = torch.where(case_5, mu_ub_inp1_prop, mu_ub)
        """
        (case 6) x1.ub <= 0 and (x2.lb < 0 and x2.ub > 0)
		Dy = -y2
		Dy.lb = -lx2.ub - m.ub
		Dy.ub = -lx2.lb
        """
        case_6 = inp1_negative & inp2_unstable
        lambda_lb_inp2 = torch.where(case_6, -lambda_ub_inp2_prop, lambda_lb_inp2)
        mu_lb = torch.where(case_6, -mu_ub_inp2_prop, mu_lb)
        lambda_ub_inp2 = torch.where(case_6, -lambda_lb_inp2_prop, lambda_ub_inp2)
        """
        (case 7) (x1.lb < 0 and x1.ub > 0) and x2.lb >= 0
		Dy = y1 - x2
		Dy.lb = Dx
		Dy.ub = lx1.ub + m.ub - x2
        """
        case_7 = inp1_unstable & inp2_positive
        lambda_lb = torch.where(case_7, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb)
        mu_lb = torch.where(case_7, torch.zeros(inp1_lb_layer.size(), device=self.device), mu_lb)
        lambda_ub_inp1 = torch.where(case_7, lambda_ub_inp1_prop, lambda_ub_inp1)
        mu_ub = torch.where(case_7, mu_ub_inp1_prop, mu_ub)
        lambda_ub_inp2 = torch.where(case_7, -torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub_inp2)
        """
        (case 8) x1.lb >= 0 and (x2.lb < 0 and x2.ub > 0)
		Dy = x1 - y2
		Dy.lb = x1 - lx2.ub - m.ub
		Dy.ub = Dx
        """
        case_8 = inp1_positive & inp2_unstable
        lambda_ub = torch.where(case_8, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_ub)
        mu_ub = torch.where(case_8, torch.zeros(inp1_lb_layer.size(), device=self.device), mu_ub)
        lambda_lb_inp1 = torch.where(case_8, torch.ones(inp1_lb_layer.size(), device=self.device), lambda_lb_inp1)
        lambda_lb_inp2 = torch.where(case_8, -lambda_ub_inp2_prop, lambda_lb_inp2)
        mu_lb = torch.where(case_8, -mu_ub_inp2_prop, mu_lb)
        """
        (case 9) (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb >= 0)
        Dy.lb = 0
        Dy.ub = y1 (x1U < dU) or Dx (x1U >= dU)
        """
        case_9 = inp1_unstable & inp2_unstable & delta_positive
        lambda_lb = torch.where(case_9, torch.zeros(inp1_lb_layer.size(), device=self.device), lambda_lb)
        mu_lb = torch.where(case_9, torch.zeros(inp1_lb_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(case_9, torch.zeros(inp1_lb_layer.size(), device=self.device), mu_ub)
        temp_lambda = torch.where((inp1_ub_layer < d_ub_layer) & case_9, lambda_ub_inp1_prop, torch.ones(inp1_lb_layer.size(), device=self.device))
        lambda_ub = torch.where(case_9 & (inp1_ub_layer >= d_ub_layer), temp_lambda, lambda_ub)
        lambda_ub_inp1 = torch.where(case_9 & (inp1_ub_layer < d_ub_layer), temp_lambda, lambda_ub_inp1)
        mu_ub = torch.where(case_9 & (inp1_ub_layer < d_ub_layer), mu_ub_inp1_prop, mu_ub)
        """
        (case 10) (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb <= 0)
        Dy.lb = Dx or y1 - y2
        Dy.ub = 0
        """
        case_10 = (inp1_unstable & inp2_unstable & delta_negative)
        lambda_ub = torch.where(case_10, torch.zeros(inp1_lb_layer.size(), device=self.device), lambda_ub)
        mu_lb = torch.where(case_10, torch.zeros(inp1_lb_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(case_10, torch.zeros(inp1_lb_layer.size(), device=self.device), mu_ub)
        temp_lambda = torch.where((inp2_ub_layer < d_ub_layer) & case_10, -lambda_ub_inp2_prop, torch.ones(inp1_lb_layer.size(), device=self.device))
        lambda_lb = torch.where((inp2_ub_layer >= d_ub_layer) & case_10, temp_lambda, lambda_lb)
        lambda_lb_inp2 = torch.where((inp2_ub_layer < d_ub_layer) & case_10, temp_lambda, lambda_lb_inp2)
        mu_lb = torch.where((inp2_ub_layer < d_ub_layer) & case_10, -mu_ub_inp2_prop, mu_lb)

        """
        (case 11) (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb < 0 and delta_ub > 0)
        """
        temp_mu = (d_lb_layer * d_ub_layer) / (d_ub_layer - d_lb_layer + 1e-15)
        temp_lambda_lb = (-d_lb_layer) / (d_ub_layer - d_lb_layer + 1e-15)
        temp_lambda_ub = d_ub_layer / (d_ub_layer - d_lb_layer + 1e-15)
        use_delta_lb = (torch.abs(temp_mu) == torch.abs(mu_ub_inp2_prop))
        use_delta_ub = (torch.abs(temp_mu) == torch.abs(mu_ub_inp1_prop))
        case_11 = (inp1_unstable & inp2_unstable & delta_unstable)

        lambda_lb = torch.where(case_11 & use_delta_lb, temp_lambda_lb, lambda_lb)
        lambda_ub = torch.where(case_11 & use_delta_ub, temp_lambda_ub, lambda_ub)
        mu_lb = torch.where(case_11 & use_delta_lb, temp_mu, mu_lb)
        mu_ub = torch.where(case_11 & use_delta_ub, -temp_mu, mu_ub)
        # lambda_lb = torch.where(case_11, temp_lambda_lb, lambda_lb)
        # lambda_ub = torch.where(case_11, temp_lambda_ub, lambda_ub)
        # mu_lb = torch.where(case_11, temp_mu, mu_lb)
        # mu_ub = torch.where(case_11, -temp_mu, mu_ub)

        lambda_lb_inp1 = torch.where(case_11 & ~use_delta_lb, lambda_lb_inp1_prop, lambda_lb_inp1)
        lambda_ub_inp1 = torch.where(case_11 & ~use_delta_ub, lambda_ub_inp1_prop, lambda_ub_inp1)
        mu_ub = torch.where(case_11 & ~use_delta_ub, mu_ub_inp1_prop, mu_ub)
        lambda_lb_inp2 = torch.where(case_11 & ~use_delta_lb, -lambda_ub_inp2_prop, lambda_lb_inp2)
        lambda_ub_inp2 = torch.where(case_11 & ~use_delta_ub, -lambda_lb_inp2_prop, lambda_ub_inp2)
        mu_lb = torch.where(case_11 & ~use_delta_lb, -mu_ub_inp2_prop, mu_lb)

        # Segregate the +ve and -ve components of the coefficients
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.d_C_lb)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.d_C_ub)
        neg_comp_lb_inp1, pos_comp_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_lb)
        neg_comp_ub_inp1, pos_comp_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.d_x1_C_ub)
        neg_comp_lb_inp2, pos_comp_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_lb)
        neg_comp_ub_inp2, pos_comp_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.d_x2_C_ub)

        neg_coef_lb_inp1, pos_coef_lb_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.x1_C_lb)
        neg_coef_ub_inp1, pos_coef_ub_inp1 = self.pos_neg_weight_decomposition(back_prop_struct.x1_C_ub)
        neg_coef_lb_inp2, pos_coef_lb_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.x2_C_lb)
        neg_coef_ub_inp2, pos_coef_ub_inp2 = self.pos_neg_weight_decomposition(back_prop_struct.x2_C_ub)

        d_C_lb = pos_comp_lb * lambda_lb + neg_comp_lb * lambda_ub
        d_x1_C_lb = pos_comp_lb_inp1 * lambda_lb_inp1_prop + neg_comp_lb_inp1 * lambda_ub_inp1_prop
        d_x1_C_lb = d_x1_C_lb + pos_comp_lb * lambda_lb_inp1 + neg_comp_lb * lambda_ub_inp1
        d_x2_C_lb = pos_comp_lb_inp2 * lambda_lb_inp2_prop + neg_comp_lb_inp2 * lambda_ub_inp2_prop
        d_x2_C_lb = d_x2_C_lb + pos_comp_lb * lambda_lb_inp2 + neg_comp_lb * lambda_ub_inp2

        d_C_ub = pos_comp_ub * lambda_ub + neg_comp_ub * lambda_lb
        d_x1_C_ub = pos_comp_ub_inp1 * lambda_ub_inp1_prop + neg_comp_ub_inp1 * lambda_lb_inp1_prop
        d_x1_C_ub = d_x1_C_ub + pos_comp_ub * lambda_ub_inp1 + neg_comp_ub * lambda_lb_inp1
        d_x2_C_ub = pos_comp_ub_inp2 * lambda_ub_inp2_prop + neg_comp_ub_inp2 * lambda_lb_inp2_prop
        d_x2_C_ub = d_x2_C_ub + pos_comp_ub * lambda_ub_inp2 + neg_comp_ub * lambda_lb_inp2

        d_b_lb = pos_comp_lb @ mu_lb + neg_comp_lb @ mu_ub + back_prop_struct.d_b_lb
        d_b_lb = d_b_lb + neg_comp_lb_inp1 @ mu_ub_inp1_prop + neg_comp_lb_inp2 @ mu_ub_inp2_prop
        d_b_ub = pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb + back_prop_struct.d_b_ub
        d_b_ub = d_b_ub + pos_comp_ub_inp1 @ mu_ub_inp1_prop + pos_comp_ub_inp2 @ mu_ub_inp2_prop

        if back_prop_struct.x1_b_lb is not None:
            x1_b_lb = back_prop_struct.x1_b_lb + neg_coef_lb_inp1 @ mu_ub_inp1_prop
            x1_b_ub = back_prop_struct.x1_b_ub + pos_coef_ub_inp1 @ mu_ub_inp1_prop
            x2_b_lb = back_prop_struct.x2_b_lb + neg_coef_lb_inp2 @ mu_ub_inp2_prop
            x2_b_ub = back_prop_struct.x2_b_ub + pos_coef_ub_inp2 @ mu_ub_inp2_prop

            x1_C_lb = neg_coef_lb_inp1 * lambda_ub_inp1_prop + pos_coef_lb_inp1 * lambda_lb_inp1_prop
            x1_C_ub = neg_coef_ub_inp1 * lambda_lb_inp1_prop + pos_coef_ub_inp1 * lambda_ub_inp1_prop
            x2_C_lb = neg_coef_lb_inp2 * lambda_ub_inp2_prop + pos_coef_lb_inp2 * lambda_lb_inp2_prop
            x2_C_ub = neg_coef_ub_inp2 * lambda_lb_inp2_prop + pos_coef_ub_inp2 * lambda_ub_inp2_prop
        else:
            x1_b_lb = None
            x1_b_ub = None
            x2_b_lb = None
            x2_b_ub = None

            x1_C_lb = None
            x1_C_ub = None
            x2_C_lb = None
            x2_C_ub = None

        back_prop_struct.populate(d_C_lb=d_C_lb, d_b_lb=d_b_lb,
                                  d_C_ub=d_C_ub, d_b_ub=d_b_ub,
                                  d_x1_C_lb=d_x1_C_lb,
                                  d_x1_C_ub=d_x1_C_ub,
                                  d_x2_C_lb=d_x2_C_lb,
                                  d_x2_C_ub=d_x2_C_ub,
                                  x1_C_lb=x1_C_lb, x2_C_lb=x2_C_lb,
                                  x1_C_ub=x1_C_ub, x2_C_ub=x2_C_ub,
                                  x1_b_lb=x1_b_lb, x2_b_lb=x2_b_lb,
                                  x1_b_ub=x1_b_ub, x2_b_ub=x2_b_ub)

        return back_prop_struct
