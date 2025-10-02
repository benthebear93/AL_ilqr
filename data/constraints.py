from dataclasses import dataclass
import jax.numpy as jnp
from src.constraints import constraint_eval


@dataclass
class ConstraintsData:
    constraints: list
    violations: list
    jacobian_state: list
    jacobian_action: list


def constraint_data_func(model, constraints):
    """
    Initialize ConstraintsData
    model: list of Dynamics
    constraints: list of Constraint
    """
    H = len(constraints)
    c = [jnp.zeros((constraints[t].num_constraint,)) for t in range(H)]
    cx = [
        jnp.zeros(
            (
                constraints[t].num_constraint,
                model[t].num_state if t < H - 1 else model[-1].num_next_state,
            )
        )
        for t in range(H)
    ]
    cu = [
        jnp.zeros((constraints[t].num_constraint, model[t].num_action))
        for t in range(H - 1)
    ]
    return ConstraintsData(constraints, c, cx, cu)


def constraint(constraint_data: ConstraintsData, x, u, w):
    """
    Evaluate constraints and update violations in constraint_data
    Equivalent to Julia's `constraint!(constraint_data, x, u, w)`
    """
    constraint_eval(constraint_data.violations, constraint_data.constraints, x, u, w)


def constraint_violation(constraint_data: ConstraintsData, norm_type=jnp.inf):
    """
    Compute maximum constraint violation given cached violations
    """
    constraints = constraint_data.constraints
    H = len(constraints)
    max_violation = 0.0
    for t in range(H):
        num_constraint = constraints[t].num_constraint
        ineq = constraints[t].indices_inequality
        for i in range(num_constraint):
            c_val = constraint_data.violations[t][i]
            cti = jnp.maximum(0.0, c_val) if i in ineq else jnp.abs(c_val)
            max_violation = jnp.maximum(max_violation, cti)
    return max_violation


def constraint_violation_eval(
    constraint_data: ConstraintsData, x, u, w, norm_type=jnp.inf
):
    """
    Evaluate constraints first, then compute violation
    """
    constraint(constraint_data, x, u, w)
    return constraint_violation(constraint_data, norm_type=norm_type)
