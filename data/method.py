import numpy as np
from .constraints import constraint_violation_eval
from src.augmented_lagrangian import cost_augmented
from src.costs import cost


def cost_method(problem, mode="nominal"):
    """
    Compute cost for a given trajectory.
    """
    if mode == "nominal":
        return cost(
            problem.objective.costs,
            problem.nominal_states,
            problem.nominal_actions,
            problem.parameters,
        )
    elif mode == "current":
        return cost(
            problem.objective.costs, problem.states, problem.actions, problem.parameters
        )
    else:
        return 0.0


def cost_update(data, problem, mode="nominal"):
    """
    Update data.objective with current or nominal trajectory cost.
    Also compute constraint violation if AugmentedLagrangian is used.
    """
    if mode == "nominal":
        J = cost_augmented(
            problem.objective.costs,
            problem.nominal_states,
            problem.nominal_actions,
            problem.parameters,
        )
    elif mode == "current":
        J = cost_augmented(
            problem.objective.costs, problem.states, problem.actions, problem.parameters
        )
    else:
        J = 0.0

    # update objective
    data.objective[0] = J

    # update constraint violation if applicable
    if hasattr(problem.objective.costs, "constraint_data"):
        violation = constraint_violation_eval(
            problem.objective.costs.constraint_data,
            problem.states,
            problem.actions,
            problem.parameters,
            norm_type=np.inf,
        )
        data.max_violation[0] = violation

    return data


def update_nominal_trajectory(problem):
    """
    Copy current states/actions into nominal trajectory.
    """
    H = len(problem.states)
    for t in range(H):
        problem.nominal_states[t] = problem.states[t].copy()
        if t < H - 1:
            problem.nominal_actions[t] = problem.actions[t].copy()
    return problem


def trajectory_sensitivities(problem, policy, data):
    """
    Compute trajectory sensitivities for line search.
    """
    H = len(problem.states)
    trajectory = np.zeros_like(problem.trajectory)

    for t in range(H - 1):
        idx_x = data.indices_state[t]
        idx_u = data.indices_action[t]
        idx_y = data.indices_state[t + 1]

        zx = trajectory[idx_x]  # state slice
        zu = policy.k[t] + policy.K[t] @ zx
        zy = (
            problem.model.jacobian_action[t] @ zu + problem.model.jacobian_state[t] @ zx
        )

        # update trajectory vector
        trajectory[idx_u] = zu
        trajectory[idx_y] = zy

    problem.trajectory = trajectory
    return problem


def trajectories(problem, mode="nominal"):
    """
    Return (states, actions, parameters) depending on mode.
    """
    if mode == "nominal":
        return problem.nominal_states, problem.nominal_actions, problem.parameters
    else:
        return problem.states, problem.actions, problem.parameters
