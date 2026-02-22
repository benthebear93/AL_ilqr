import numpy as np
from .rollout import rollout_with_policy_inplace
from .lagrangian import lagrangian_gradient
from data.method import (
    trajectory_sensitivities,
    cost_update,
    update_nominal_trajectory,
)


def forward_pass(
    policy,
    problem,
    data,
    line_search="armijo",
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=25,
    verbose=False,
):
    data.status[0] = False
    J_prev = data.objective[0]
    data = lagrangian_gradient(data, policy, problem)

    if line_search == "armijo":
        problem = trajectory_sensitivities(problem, policy, data)
        delta_grad_product = data.gradient @ problem.trajectory
    else:
        delta_grad_product = 0.0

    step_size = 1.0
    iteration = 1

    while step_size >= min_step_size:
        if iteration > max_iterations:
            if verbose:
                print("[warn] forward pass failure (max iterations)")
            break

        J = np.inf

        rollout_with_policy_inplace(policy, problem, step_size=step_size)
        data = cost_update(data, problem, mode="current")
        J = data.objective[0]

        if J <= J_prev + c1 * step_size * delta_grad_product:
            update_nominal_trajectory(problem)
            data.objective[0] = J
            data.status[0] = True
            data.step_size[0] = step_size
            break
        else:
            step_size *= 0.5
            iteration += 1
            data.step_size[0] = step_size

    if step_size < min_step_size and verbose:
        print("[warn] line search failure")

    return data
