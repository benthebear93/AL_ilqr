from dataclasses import dataclass
import jax.numpy as jnp

from data.policy import policy_data, PolicyData
from data.problem import problem_data, ProblemData
from data.solver import solver_data, SolverData
from .augmented_lagrangian import augmented_lagrangian
from .options import Options


@dataclass
class Solver:
    problem: ProblemData
    policy: PolicyData
    data: SolverData
    options: object


def solver_from_objective(dynamics, objective, parameters=None, options=None):
    if parameters is None:
        parameters = [jnp.zeros(d.num_parameter) for d in dynamics] + [jnp.zeros((0,))]
    if options is None:
        options = Options()

    # allocate policy data
    policy = policy_data(dynamics)

    # allocate problem data
    problem = problem_data(dynamics, objective, parameters=parameters)

    # allocate solver data
    data = solver_data(dynamics)

    return Solver(problem=problem, policy=policy, data=data, options=options)


def solver_from_costs_constraints(
    dynamics, costs, constraints, parameters=None, options=None
):
    if parameters is None:
        parameters = [jnp.zeros(d.num_parameter) for d in dynamics] + [jnp.zeros((0,))]
    if options is None:
        options = Options()

    objective_al = augmented_lagrangian(dynamics, costs, constraints)
    policy = policy_data(dynamics)
    problem = problem_data(dynamics, objective_al, parameters=parameters)
    data = solver_data(dynamics)

    return Solver(problem=problem, policy=policy, data=data, options=options)


def get_trajectory(solver: Solver):
    """Return nominal states and actions (excluding last)"""
    return solver.problem.nominal_states, solver.problem.nominal_actions[:-1]


def current_trajectory(solver: Solver):
    """Return current states and actions (excluding last)"""
    return solver.problem.states, solver.problem.actions[:-1]


def initialize_controls(solver: Solver, actions):
    for t, ut in enumerate(actions):
        solver.problem.nominal_actions[t] = ut


def initialize_states(solver: Solver, states):
    for t, xt in enumerate(states):
        solver.problem.nominal_states[t] = xt
