from dataclasses import dataclass
import numpy as np


# ================================================
# Value function approximation
# ================================================
@dataclass
class Value:
    gradient: list  # list of (num_state,) arrays
    hessian: list  # list of (num_state, num_state) arrays


# ================================================
# Action-value function approximation
# ================================================
@dataclass
class ActionValue:
    gradient_state: list  # list of (num_state,) arrays
    gradient_action: list  # list of (num_action,) arrays
    hessian_state_state: list  # list of (num_state, num_state)
    hessian_action_action: list  # list of (num_action, num_action)
    hessian_action_state: list  # list of (num_action, num_state)


# ================================================
# Policy Data
# ================================================
@dataclass
class PolicyData:
    # policy u = ū + K * (x - x̄) + k
    K: list
    k: list
    K_candidate: list
    k_candidate: list

    value: Value
    action_value: ActionValue

    # pre-allocated memory
    xx_hat_tmp: list
    ux_hat_tmp: list
    uu_tmp: list
    ux_tmp: list


def policy_data(dynamics):
    """
    dynamics: list of Dynamics objects
    return: PolicyData object
    """

    # feedback & feedforward gains
    K = [np.zeros((d.num_action, d.num_state)) for d in dynamics]
    k = [np.zeros((d.num_action,)) for d in dynamics]

    K_candidate = [np.zeros((d.num_action, d.num_state)) for d in dynamics]
    k_candidate = [np.zeros((d.num_action,)) for d in dynamics]

    # value function approximation
    P = [np.zeros((d.num_state, d.num_state)) for d in dynamics]
    p = [np.zeros((d.num_state,)) for d in dynamics]

    # add terminal state dimension
    P.append(np.zeros((dynamics[-1].num_next_state, dynamics[-1].num_next_state)))
    p.append(np.zeros((dynamics[-1].num_next_state,)))

    value = Value(p, P)

    # action-value function approximation
    Qx = [np.zeros((d.num_state,)) for d in dynamics]
    Qu = [np.zeros((d.num_action,)) for d in dynamics]
    Qxx = [np.zeros((d.num_state, d.num_state)) for d in dynamics]
    Quu = [np.zeros((d.num_action, d.num_action)) for d in dynamics]
    Qux = [np.zeros((d.num_action, d.num_state)) for d in dynamics]

    action_value = ActionValue(Qx, Qu, Qxx, Quu, Qux)

    # pre-allocated memory
    xx_hat_tmp = [np.zeros((d.num_state, d.num_next_state)) for d in dynamics]
    ux_hat_tmp = [np.zeros((d.num_action, d.num_next_state)) for d in dynamics]
    uu_tmp = [np.zeros((d.num_action, d.num_action)) for d in dynamics]
    ux_tmp = [np.zeros((d.num_action, d.num_state)) for d in dynamics]

    return PolicyData(
        K,
        k,
        K_candidate,
        k_candidate,
        value,
        action_value,
        xx_hat_tmp,
        ux_hat_tmp,
        uu_tmp,
        ux_tmp,
    )
