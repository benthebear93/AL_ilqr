from dataclasses import dataclass
import numpy as np

from .model import model_data
from .objective import objective_data, ObjectiveData
from src.dynamics import num_trajectory
from src.augmented_lagrangian import AugmentedLagrangianCosts
from .costs import CostCollection


@dataclass
class ProblemData:
    # current trajectory
    states: list
    actions: list

    # disturbance trajectory
    parameters: list

    # nominal trajectory
    nominal_states: list
    nominal_actions: list

    # model data
    model: object

    # objective data (ObjectiveData or AugmentedLagrangianCosts)
    objective: ObjectiveData

    # trajectory vector z
    trajectory: object


def problem_data(dynamics, costs, parameters=None):
    """
    Python port of Julia's problem_data
    dynamics: list of Dynamics objects
    costs: list of Cost objects OR AugmentedLagrangianCosts
    parameters: list of disturbance vectors (default zeros)
    """
    if parameters is None:
        parameters = [np.zeros(d.num_parameter) for d in dynamics] + [np.zeros((0,))]

    if len(parameters) == len(dynamics):
        parameters = parameters + [np.zeros((0,))]
    assert len(dynamics) + 1 == len(parameters)

    # current trajectory
    states = [np.zeros((d.num_state,)) for d in dynamics] + [
        np.zeros((dynamics[-1].num_next_state,))
    ]
    actions = [np.zeros((d.num_action,)) for d in dynamics] + [np.zeros((0,))]

    # nominal trajectory
    nominal_states = [np.zeros((d.num_state,)) for d in dynamics] + [
        np.zeros((dynamics[-1].num_next_state,))
    ]
    nominal_actions = [np.zeros((d.num_action,)) for d in dynamics] + [np.zeros((0,))]

    # model data
    model = model_data(dynamics)

    objective = objective_data(dynamics, costs)
    # trajectory vector z
    trajectory = np.zeros(num_trajectory(dynamics))

    return ProblemData(
        states,
        actions,
        parameters,
        nominal_states,
        nominal_actions,
        model,
        objective,
        trajectory,
    )
