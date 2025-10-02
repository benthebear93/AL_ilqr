from dataclasses import replace
import jax
import jax.numpy as jnp

from data.problem import ProblemData
from data.policy import PolicyData
from data.solver import SolverData
from data.model import reset_model
from data.objective import reset_objective
from .solver import Solver
from src.backward_pass import backward_pass
from src.forward_pass import forward_pass
from data.method import cost, cost_update
from .gradient import compute_problem_gradients
from .augmented_lagrangian import augmented_lagrangian_update, AugmentedLagrangianCosts
from .lagrangian import lagrangian_gradient


def ilqr_solve(solver: Solver, iteration=1):
    if solver.options.verbose and iteration == 1:
        print("solver_info()")  # placeholder

    data = solver.data
    problem = solver.problem
    reset_model(problem.model)
    reset_objective(problem.objective)
    policy = solver.policy

    data = cost_update(data, problem, mode="nominal")
    problem = compute_problem_gradients(problem, mode="nominal")
    policy, problem = backward_pass(policy, problem, mode="nominal")

    obj_prev = data.objective[0]
    for i in range(solver.options.max_iterations):
        data = forward_pass(
            policy,
            problem,
            data,
            min_step_size=solver.options.min_step_size,
            line_search=solver.options.line_search,
            verbose=solver.options.verbose,
        )
        if solver.options.line_search != "none":
            problem = compute_problem_gradients(problem, mode="nominal")
            policy, problem = backward_pass(policy, problem, mode="nominal")
            data = lagrangian_gradient(data, policy, problem)

        gradient_norm = jnp.linalg.norm(data.gradient, jnp.inf)

        iterations = data.iterations.at[0].add(1)
        data = replace(data, iterations=iterations)

        if solver.options.verbose:
            print(
                f"""iter: {i}
                cost: {data.objective[0]}
                gradient_norm: {gradient_norm}
                max_violation: {data.max_violation[0]}
                step_size: {data.step_size[0]}"""
            )

        # convergence check
        if gradient_norm < solver.options.lagrangian_gradient_tolerance:
            break
        if abs(data.objective[0] - obj_prev) < solver.options.objective_tolerance:
            break
        else:
            obj_prev = data.objective[0]

        if not data.status[0]:
            break

    return replace(solver, data=data)


#############################
# Augmented Lagrangian Solver
#############################
def constrained_ilqr_solve(
    solver: Solver, augmented_lagrangian_callback=lambda x: None
):
    data = solver.data
    objective = solver.problem.objective
    if not isinstance(objective.costs, AugmentedLagrangianCosts):
        raise TypeError(
            "constrained_ilqr_solve requires AugmentedLagrangianCosts objective"
        )

    # reset duals, penalties
    for lam in objective.costs.constraint_dual:
        lam = lam * 0.0
    for rho in objective.costs.constraint_penalty:
        rho = rho * solver.options.initial_constraint_penalty

    for i in range(solver.options.max_dual_updates):
        if solver.options.verbose:
            print(f"  al iter: {i}")

        # primal minimization
        solver = ilqr_solve(solver, iteration=i)

        # update trajectories
        data = cost_update(solver.data, solver.problem, mode="nominal")

        # constraint check
        if data.max_violation[0] <= solver.options.constraint_tolerance:
            break

        # dual ascent
        augmented_lagrangian_update(
            objective.costs,
            scaling_penalty=solver.options.scaling_penalty,
            max_penalty=solver.options.max_penalty,
        )

        # user-defined callback
        augmented_lagrangian_callback(solver)

    return solver


def solve(solver, *args, **kwargs):
    if isinstance(solver.problem.objective.costs, AugmentedLagrangianCosts):
        return constrained_ilqr_solve(solver, *args, **kwargs)
    else:
        return ilqr_solve(solver, *args, **kwargs)
