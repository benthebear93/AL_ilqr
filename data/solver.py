from dataclasses import dataclass
import numpy as np
from src.dynamics import num_trajectory


@dataclass
class SolverData:
    objective: np.ndarray  # shape (1,)
    gradient: np.ndarray  # shape (n_total + m_total,)
    max_violation: np.ndarray  # shape (1,)
    step_size: np.ndarray  # shape (1,)
    iterations: np.ndarray  # shape (1,), int
    status: np.ndarray  # shape (1,), bool
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
        indices_state.append(np.arange(n_sum, n_sum + d.num_state))
        indices_action.append(
            np.arange(n_total + m_sum, n_total + m_sum + d.num_action)
        )
        n_sum += d.num_state
        m_sum += d.num_action

    # last terminal state indices
    indices_state.append(np.arange(n_sum, n_sum + dynamics[-1].num_next_state))

    objective = np.zeros(1)
    max_violation = np.zeros(1)
    step_size = np.zeros(1)
    gradient = np.zeros(num_trajectory(dynamics))
    iterations = np.zeros(1, dtype=int)  # like data.iterations[1] in Julia
    status = np.ones(1, dtype=bool)  # like data.status[1] in Julia (true/false)

    cache = {
        "objective": np.zeros(max_cache),
        "gradient": np.zeros((max_cache, gradient.shape[0])),  # history of gradients
        "max_violation": np.zeros(max_cache),
        "step_size": np.zeros(max_cache),
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
    data.gradient = np.zeros_like(data.gradient)
    data.max_violation[0] = 0.0
    data.cache["objective"] = np.zeros_like(data.cache["objective"])
    data.cache["gradient"] = np.zeros_like(data.cache["gradient"])
    data.cache["max_violation"] = np.zeros_like(data.cache["max_violation"])
    data.cache["step_size"] = np.zeros_like(data.cache["step_size"])
    data.status[0] = False
    data.iterations[0] = 0


def cache_update(data: SolverData, iter=0):
    """
    Update solver cache at iteration index
    """
    # NOTE: in Julia, iter handling was TODO. Here we pass explicit iter index.
    if iter >= len(data.cache["objective"]):
        raise ValueError("Solver data cache exceeded")
    data.cache["objective"][iter] = data.objective[0]
    data.cache["gradient"][iter] = np.linalg.norm(data.gradient)
    data.cache["max_violation"][iter] = data.max_violation[0]
    data.cache["step_size"][iter] = data.step_size[0]
