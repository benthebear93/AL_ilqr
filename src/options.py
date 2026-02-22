from dataclasses import dataclass
import numpy as np


@dataclass
class Options:
    line_search: str = "armijo"  # Julia: Symbol(:armijo)
    max_iterations: int = 100
    max_dual_updates: int = 10
    min_step_size: float = 1.0e-5
    objective_tolerance: float = 1.0e-3
    lagrangian_gradient_tolerance: float = 1.0e-3
    constraint_tolerance: float = 5.0e-3
    constraint_norm: float = np.inf  # Julia: Inf
    initial_constraint_penalty: float = 1.0
    scaling_penalty: float = 10.0
    max_penalty: float = 1.0e8
    reset_cache: bool = False
    verbose: bool = True
