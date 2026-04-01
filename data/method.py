import numpy as np
from .constraints import constraint_violation_eval
from src.augmented_lagrangian import AugmentedLagrangianCosts, cost_augmented
from src.costs import cost


def cost_method(problem, mode="nominal"):
    """
    Compute cost for a given trajectory.
    """
    objective = problem.objective.costs
    if mode == "nominal":
        states = problem.nominal_states
        actions = problem.nominal_actions
    elif mode == "current":
        states = problem.states
        actions = problem.actions
    else:
        return 0.0

    if isinstance(objective, AugmentedLagrangianCosts):
        return cost_augmented(objective, states, actions, problem.parameters)
    return cost(objective, states, actions, problem.parameters)


def cost_update(data, problem, mode="nominal"):
    """
    Update data.objective with current or nominal trajectory cost.
    Also compute constraint violation if AugmentedLagrangian is used.
    """
    objective = problem.objective.costs
    if mode == "nominal":
        states = problem.nominal_states
        actions = problem.nominal_actions
    elif mode == "current":
        states = problem.states
        actions = problem.actions
    else:
        J = 0.0

    if mode in ("nominal", "current"):
        if isinstance(objective, AugmentedLagrangianCosts):
            J = cost_augmented(objective, states, actions, problem.parameters)
        else:
            J = cost(objective, states, actions, problem.parameters)

    # update objective
    data.objective[0] = J

    # update constraint violation if applicable
    if isinstance(objective, AugmentedLagrangianCosts):
        violation = constraint_violation_eval(
            objective.constraint_data,
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
