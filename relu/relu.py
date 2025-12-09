from enum import Enum


class ReluCondition(Enum):
    UNINITIALIZED = 0
    POSITIVE = 1
    NEGATIVE = 2
    UNSTABLE_0 = 3  # triangle with lower bound is y = 0
    UNSTABLE_1 = 4  # triangle with lower bound is y = x


class ReluApprox(Enum):
    TRIANGLE_0 = 3  # triangle with lower bound is y = 0
    TRIANGLE_1 = 4  # triangle with lower bound is y = x


class ReluInput:
    def __init__(self, condition=None, approx=None):
        self.condition = condition
        self.approx = approx
