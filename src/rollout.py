import numpy as np


def rollout(dynamics, initial_state, actions):
    x_hist = [np.asarray(initial_state, dtype=float)]
    x_prev = x_hist[0]
    for u in actions:
        x_prev = dynamics.evaluate(x_prev, u)
        x_hist.append(np.asarray(x_prev, dtype=float))
    return np.vstack(x_hist)


def rollout_with_policy_inplace(policy, problem, step_size=1.0):
    """
    Rollout trajectory using current policy (in-place update of problem states/actions).
    Equivalent to Julia's rollout!.
    """
    dynamics = problem.model.dynamics
    x = problem.states
    u = problem.actions
    w = problem.parameters
    x_bar = problem.nominal_states
    u_bar = problem.nominal_actions

    K = policy.K
    k = policy.k

    x[0] = x_bar[0]

    # rollout
    for t, d in enumerate(dynamics):
        u[t] = u_bar[t] + K[t] @ (x[t] - x_bar[t]) + step_size * k[t]
        x[t + 1] = d.evaluate(x[t], u[t], w[t])
