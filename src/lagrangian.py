from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
from data.problem import ProblemData
from data.policy import PolicyData
from data.solver import SolverData


def lagrangian_gradient(data: SolverData, policy: PolicyData, problem: ProblemData):
    """
    Compute Lagrangian gradient wrt state and action
    """
    p = policy.value.gradient
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    H = len(problem.states)

    grad = data.gradient.copy()

    for t in range(H - 1):
        idx_x = data.indices_state[t]
        idx_u = data.indices_action[t]

        grad = grad.at[idx_x].set(Qx[t] - p[t])
        grad = grad.at[idx_u].set(Qu[t])
    return replace(data, gradient=grad)
