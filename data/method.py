import jax.numpy as jnp
from dataclasses import replace
from .constraints import constraint_violation, constraint_violation_eval
from src.costs import cost
from src.augmented_lagrangian import cost_augmented


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
    data = replace(data, objective=data.objective.at[0].set(J))

    # update constraint violation if applicable
    if hasattr(problem.objective.costs, "constraint_data"):
        violation = constraint_violation_eval(
            problem.objective.costs.constraint_data,
            problem.states,
            problem.actions,
            problem.parameters,
            norm_type=jnp.inf,
        )
        data = replace(data, max_violation=data.max_violation.at[0].set(violation))

    return data


def update_nominal_trajectory(problem):
    """
    Copy current states/actions into nominal trajectory.
    """
    H = len(problem.states)
    new_nominal_states = list(problem.nominal_states)
    new_nominal_actions = list(problem.nominal_actions)

    for t in range(H):
        problem.nominal_states[t] = problem.states[t]
        if t < H - 1:
            problem.nominal_actions[t] = problem.actions[t]
    return replace(
        problem,
        nominal_states=new_nominal_states,
        nominal_actions=new_nominal_actions,
    )


def trajectory_sensitivities(problem, policy, data):
    """
    Compute trajectory sensitivities for line search.
    """
    H = len(problem.states)
    trajectory = jnp.zeros_like(problem.trajectory)

    for t in range(H - 1):
        idx_x = data.indices_state[t]
        idx_u = data.indices_action[t]
        idx_y = data.indices_state[t + 1]

        zx = trajectory.at[idx_x].get()  # state slice
        zu = policy.k[t] + policy.K[t] @ zx
        zy = (
            problem.model.jacobian_action[t] @ zu + problem.model.jacobian_state[t] @ zx
        )

        # update trajectory vector
        trajectory = trajectory.at[idx_u].set(zu)
        trajectory = trajectory.at[idx_y].set(zy)

    return replace(problem, trajectory=trajectory)


def trajectories(problem, mode="nominal"):
    """
    Return (states, actions, parameters) depending on mode.
    """
    if mode == "nominal":
        return problem.nominal_states, problem.nominal_actions, problem.parameters
    else:
        return problem.states, problem.actions, problem.parameters
