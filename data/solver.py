from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np  # for Inf
from src.dynamics import num_trajectory


@dataclass
class SolverData:
    objective: jnp.ndarray  # shape (1,)
    gradient: jnp.ndarray  # shape (n_total + m_total,)
    max_violation: jnp.ndarray  # shape (1,)
    step_size: jnp.ndarray  # shape (1,)
    iterations: jnp.ndarray  # shape (1,), int
    status: jnp.ndarray  # shape (1,), bool
    indices_state: list = None
    indices_action: list = None
    cache: dict = None


def solver_data(dynamics, max_cache=1000):
    """
    Python port of Julia's solver_data
    dynamics: list of Dynamics objects
    """
    indices_state = []
    indices_action = []
    n_sum = 0
    m_sum = 0
    n_total = sum(d.num_state for d in dynamics) + dynamics[-1].num_next_state

    for d in dynamics:
        indices_state.append(jnp.arange(n_sum, n_sum + d.num_state))
        indices_action.append(
            jnp.arange(n_total + m_sum, n_total + m_sum + d.num_action)
        )
        n_sum += d.num_state
        m_sum += d.num_action

    # last terminal state indices
    indices_state.append(jnp.arange(n_sum, n_sum + dynamics[-1].num_next_state))

    objective = jnp.zeros(1)
    max_violation = jnp.zeros(1)
    step_size = jnp.zeros(1)
    gradient = jnp.zeros(num_trajectory(dynamics))
    iterations = jnp.zeros(1, dtype=int)  # like data.iterations[1] in Julia
    status = jnp.ones(1, dtype=bool)  # like data.status[1] in Julia (true/false)

    cache = {
        "objective": jnp.zeros(max_cache),
        "gradient": jnp.zeros((max_cache, gradient.shape[0])),  # history of gradients
        "max_violation": jnp.zeros(max_cache),
        "step_size": jnp.zeros(max_cache),
    }

    return SolverData(
        objective=objective,
        gradient=gradient,
        max_violation=max_violation,
        indices_state=indices_state,
        indices_action=indices_action,
        step_size=step_size,
        status=status,
        iterations=iterations,
        cache=cache,
    )


def reset(data: SolverData):
    """
    Reset SolverData to initial state
    """
    data.objective[0] = 0.0
    data.gradient = jnp.zeros_like(data.gradient)
    data.max_violation[0] = 0.0
    data.cache["objective"] = jnp.zeros_like(data.cache["objective"])
    data.cache["gradient"] = jnp.zeros_like(data.cache["gradient"])
    data.cache["max_violation"] = jnp.zeros_like(data.cache["max_violation"])
    data.cache["step_size"] = jnp.zeros_like(data.cache["step_size"])
    data.status[0] = False
    data.iterations[0] = 0


def cache_update(data: SolverData, iter=0):
    """
    Update solver cache at iteration index
    """
    # NOTE: in Julia, iter handling was TODO. Here we pass explicit iter index.
    if iter >= len(data.cache["objective"]):
        raise ValueError("Solver data cache exceeded")
    data.cache["objective"] = data.cache["objective"].at[iter].set(data.objective[0])
    data.cache["gradient"] = (
        data.cache["gradient"].at[iter].set(jnp.linalg.norm(data.gradient))
    )
    data.cache["max_violation"] = (
        data.cache["max_violation"].at[iter].set(data.max_violation[0])
    )
    data.cache["step_size"] = data.cache["step_size"].at[iter].set(data.step_size[0])
