from dataclasses import dataclass
import numpy as np
from src.augmented_lagrangian import AugmentedLagrangianCosts


@dataclass
class ObjectiveData:
    costs: AugmentedLagrangianCosts  # CostCollection OR AugmentedLagrangianCosts
    gradient_state: list
    gradient_action: list
    hessian_state_state: list
    hessian_action_action: list
    hessian_action_state: list


def objective_data(dynamics, costs):
    """
    Wrap list of Cost objects into CostCollection automatically
    """
    gradient_state = [np.zeros((d.num_state,)) for d in dynamics] + [
        np.zeros((dynamics[-1].num_next_state,))
    ]
    gradient_action = [np.zeros((d.num_action,)) for d in dynamics]

    hessian_state_state = [np.zeros((d.num_state, d.num_state)) for d in dynamics] + [
        np.zeros((dynamics[-1].num_next_state, dynamics[-1].num_next_state))
    ]
    hessian_action_action = [np.zeros((d.num_action, d.num_action)) for d in dynamics]
    hessian_action_state = [np.zeros((d.num_action, d.num_state)) for d in dynamics]

    return ObjectiveData(
        costs,
        gradient_state,
        gradient_action,
        hessian_state_state,
        hessian_action_action,
        hessian_action_state,
    )


def reset_objective(obj: ObjectiveData):
    """
    Reset all cached gradients and Hessians to zero
    """
    H = len(obj.gradient_state)
    for t in range(H):
        obj.gradient_state[t].fill(0.0)
        obj.hessian_state_state[t].fill(0.0)
        if t == H - 1:
            continue
        obj.gradient_action[t].fill(0.0)
        obj.hessian_action_action[t].fill(0.0)
        obj.hessian_action_state[t].fill(0.0)
