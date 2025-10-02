from dataclasses import dataclass
import jax.numpy as jnp

from data.constraints import constraint_data_func, constraint
from src.costs import cost


@dataclass
class AugmentedLagrangianCosts:
    costs: object
    constraint_data: object
    constraint_penalty: list
    constraint_penalty_matrix: list
    constraint_dual: list
    active_set: list
    constraint_tmp: list
    constraint_jacobian_state_tmp: list
    constraint_jacobian_action_tmp: list


def augmented_lagrangian(model, costs, constraints):
    """
    Construct augmented Lagrangian problem
    model: list of Dynamics objects
    costs: list of Cost objects
    constraints: list of Constraint objects
    """
    H = len(model) + 1

    constraint_data = constraint_data_func(model, constraints)
    # penalties
    constraint_penalty = [jnp.ones(c.num_constraint) for c in constraints]
    constraint_penalty_matrix = [
        jnp.diag(jnp.ones(c.num_constraint)) for c in constraints
    ]

    # duals
    constraint_dual = [jnp.zeros(c.num_constraint) for c in constraints]

    # active set
    active_set = [jnp.ones((c.num_constraint,), dtype=int) for c in constraints]

    # pre-allocated memory
    constraint_tmp = [jnp.zeros(c.num_constraint) for c in constraints]
    constraint_jacobian_state_tmp = [
        jnp.zeros(
            (
                c.num_constraint,
                model[t].num_state if t < H - 1 else model[-1].num_next_state,
            )
        )
        for t, c in enumerate(constraints)
    ]
    constraint_jacobian_action_tmp = [
        jnp.zeros((c.num_constraint, model[t].num_action if t < H - 1 else 0))
        for t, c in enumerate(constraints)
    ]

    return AugmentedLagrangianCosts(
        costs,
        constraint_data,
        constraint_penalty,
        constraint_penalty_matrix,
        constraint_dual,
        active_set,
        constraint_tmp,
        constraint_jacobian_state_tmp,
        constraint_jacobian_action_tmp,
    )


def cost_augmented(objective: AugmentedLagrangianCosts, states, actions, parameters):
    """
    Evaluate augmented Lagrangian cost
    """
    J = cost(objective.costs, states, actions, parameters)

    c = objective.constraint_data.violations
    rho = objective.constraint_penalty
    lam = objective.constraint_dual
    a = objective.active_set

    H = len(c)

    constraint(objective.constraint_data, states, actions, parameters)
    active_set_update(a, objective.constraint_data, lam)

    for t in range(H):
        J += lam[t].T @ c[t]
        num_constraint = objective.constraint_data.constraints[t].num_constraint
        for i in range(num_constraint):
            if a[t][i] == 1:
                J += 0.5 * rho[t][i] * c[t][i] ** 2

    return J


def active_set_update(a, data, lam):
    """
    Update active set of inequality constraints
    """
    c = data.violations
    H = len(c)

    for t in range(H):
        # set all constraints active
        a[t] = jnp.ones_like(a[t])

        # check inequality constraints
        for i in data.constraints[t].indices_inequality:
            if (c[t][i] < 0.0) and (lam[t][i] == 0.0):
                a[t] = a[t].at[i].set(0)


def augmented_lagrangian_update(
    objective: AugmentedLagrangianCosts, scaling_penalty=10.0, max_penalty=1.0e12
):
    """
    Update multipliers and penalties for augmented Lagrangian
    """
    c = objective.constraint_data.violations
    constraints = objective.constraint_data.constraints
    rho = objective.constraint_penalty
    lam = objective.constraint_dual

    H = len(c)

    for t in range(H):
        num_constraint = constraints[t].num_constraint
        for i in range(num_constraint):
            lam[t] = lam[t].at[i].set(lam[t][i] + rho[t][i] * c[t][i])
            if i in constraints[t].indices_inequality:
                lam[t] = lam[t].at[i].set(jnp.maximum(0.0, lam[t][i]))
            rho[t] = (
                rho[t].at[i].set(jnp.minimum(scaling_penalty * rho[t][i], max_penalty))
            )
