from data.method import trajectories
from .costs import cost_gradient, cost_hessian
from .constraints import jacobian_const
from .augmented_lagrangian import AugmentedLagrangianCosts
from src.dynamics import jacobian_model


def compute_model_gradients(dynamics, problem, mode="nominal"):
    """
    Compute Jacobians of dynamics wrt state and action.
    dynamics : list of Dynamics objects
    """
    x, u, w = trajectories(problem, mode=mode)
    jx = problem.model.jacobian_state
    ju = problem.model.jacobian_action
    jacobian_model(jx, ju, dynamics, x, u, w)
    return problem


def compute_objective_gradients(objective, problem, mode="nominal"):
    """
    Compute cost gradient and Hessian contributions.
    """
    x, u, w = trajectories(problem, mode=mode)
    gx = problem.objective.gradient_state
    gu = problem.objective.gradient_action
    gxx = problem.objective.hessian_state_state
    guu = problem.objective.hessian_action_action
    gux = problem.objective.hessian_action_state
    cost_gradient(gx, gu, objective, x, u, w)
    cost_hessian(gxx, guu, gux, objective, x, u, w)

    return problem


def compute_augmented_lagrangian_gradients(objective, problem, mode="nominal"):
    """
    Compute gradients and Hessians for Augmented Lagrangian costs.
    """
    gx = problem.objective.gradient_state
    gu = problem.objective.gradient_action
    gxx = problem.objective.hessian_state_state
    guu = problem.objective.hessian_action_action
    gux = problem.objective.hessian_action_state

    constraints = objective.constraint_data.constraints
    c = objective.constraint_data.violations
    cx = objective.constraint_data.jacobian_state
    cu = objective.constraint_data.jacobian_action
    rho = objective.constraint_penalty
    lam = objective.constraint_dual
    a = objective.active_set
    I_rho = objective.constraint_penalty_matrix
    c_tmp = objective.constraint_tmp
    cx_tmp = objective.constraint_jacobian_state_tmp
    cu_tmp = objective.constraint_jacobian_action_tmp

    H = len(constraints)

    compute_objective_gradients(objective.costs, problem, mode=mode)
    compute_constraints_gradients(objective.constraint_data, problem, mode=mode)

    for t in range(H):
        num_constraint = constraints[t].num_constraint
        for i in range(num_constraint):
            I_rho[t][i, i] = rho[t][i] * a[t][i]

        c_tmp[t] = lam[t] + I_rho[t] @ c[t]

        gx[t] = gx[t] + cx[t].T @ c_tmp[t]

        cx_tmp[t] = I_rho[t] @ cx[t]
        gxx[t] = gxx[t] + cx[t].T @ cx_tmp[t]
        if t == H - 1:
            continue

        # gradient_action
        gu[t] = gu[t] + cu[t].T @ c_tmp[t]

        # hessian_action_action
        cu_tmp[t] = I_rho[t] @ cu[t]
        guu[t] = guu[t] + cu[t].T @ cu_tmp[t]

        # hessian_action_state
        gux[t] = gux[t] + cu[t].T @ cx_tmp[t]

    return problem


def compute_constraints_gradients(constraints_data, problem, mode="nominal"):
    """
    Compute Jacobians for constraints.
    """
    x, u, w = trajectories(problem, mode=mode)
    cx = constraints_data.jacobian_state
    cu = constraints_data.jacobian_action

    jacobian_const(cx, cu, constraints_data.constraints, x, u, w)
    return problem


def compute_problem_gradients(problem, mode="nominal"):
    """
    Compute both model and cost gradients.
    """
    compute_model_gradients(problem.model.dynamics, problem, mode=mode)
    objective = problem.objective.costs
    if isinstance(objective, AugmentedLagrangianCosts):
        compute_augmented_lagrangian_gradients(objective, problem, mode=mode)
    else:
        compute_objective_gradients(objective, problem, mode=mode)

    return problem
