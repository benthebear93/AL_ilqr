from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class ModelData:
    dynamics: list
    jacobian_state: list
    jacobian_action: list
    jacobian_parameter: list


def model_data(dynamics):
    """
    Python port of Julia's model_data
    dynamics: list of Dynamics objects
    """
    jacobian_state = [jnp.zeros((d.num_next_state, d.num_state)) for d in dynamics]
    jacobian_action = [jnp.zeros((d.num_next_state, d.num_action)) for d in dynamics]
    jacobian_parameter = [
        jnp.zeros((d.num_next_state, d.num_parameter)) for d in dynamics
    ]

    return ModelData(dynamics, jacobian_state, jacobian_action, jacobian_parameter)


def reset_model(model: ModelData):
    """
    Reset all Jacobian caches to zero
    """
    H = len(model.dynamics) + 1
    for t in range(H - 1):
        model.jacobian_state[t] = jnp.zeros_like(model.jacobian_state[t])
        model.jacobian_action[t] = jnp.zeros_like(model.jacobian_action[t])
        model.jacobian_parameter[t] = jnp.zeros_like(model.jacobian_parameter[t])
