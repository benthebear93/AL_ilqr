from dataclasses import dataclass
import jax.numpy as jnp
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
    gradient_state = [jnp.zeros((d.num_state,)) for d in dynamics] + [
        jnp.zeros((dynamics[-1].num_next_state,))
    ]
    gradient_action = [jnp.zeros((d.num_action,)) for d in dynamics]

    hessian_state_state = [jnp.zeros((d.num_state, d.num_state)) for d in dynamics] + [
        jnp.zeros((dynamics[-1].num_next_state, dynamics[-1].num_next_state))
    ]
    hessian_action_action = [jnp.zeros((d.num_action, d.num_action)) for d in dynamics]
    hessian_action_state = [jnp.zeros((d.num_action, d.num_state)) for d in dynamics]

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
        obj.gradient_state[t] = jnp.zeros_like(obj.gradient_state[t])
        obj.hessian_state_state[t] = jnp.zeros_like(obj.hessian_state_state[t])
        if t == H - 1:
            continue
        obj.gradient_action[t] = jnp.zeros_like(obj.gradient_action[t])
        obj.hessian_action_action[t] = jnp.zeros_like(obj.hessian_action_action[t])
        obj.hessian_action_state[t] = jnp.zeros_like(obj.hessian_action_state[t])
