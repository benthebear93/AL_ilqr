import numpy as np

from .finite_diff import jacobian


class Constraint:
    def __init__(
        self, f, num_state, num_action, indices_inequality=None, num_parameter=0
    ):
        """
        f: constraint function f(x, u, [w]) -> vector (shape (m,))
        num_state: dimension of state
        num_action: dimension of action
        num_parameter: optional parameter dimension
        indices_inequality: list of indices treated as inequality constraints
        """
        self.f = f
        self.num_state = num_state
        self.num_action = num_action
        self.num_parameter = num_parameter
        self.indices_inequality = indices_inequality or []

        # dimension of constraints (m)
        dummy_x = np.zeros((num_state,))
        dummy_u = np.zeros((num_action,))
        if num_parameter > 0:
            dummy_w = np.zeros((num_parameter,))
            m = np.asarray(f(dummy_x, dummy_u, dummy_w)).shape[0]
        else:
            m = np.asarray(f(dummy_x, dummy_u)).shape[0]
        self.num_constraint = m

        self.evaluate_cache = np.zeros((self.num_constraint,))
        self.jacobian_state_cache = np.zeros((self.num_constraint, num_state))
        self.jacobian_action_cache = np.zeros((self.num_constraint, num_action))

    def evaluate(self, x, u, w=None):
        out = self.f(x, u, w) if self.num_parameter > 0 else self.f(x, u)
        self.evaluate_cache = out
        return self.evaluate_cache

    def jacobian_state(self, x, u, w=None):
        x = np.asarray(x, dtype=float)
        if self.num_parameter > 0:
            out = jacobian(lambda xx: self.f(xx, u, w), x)
        else:
            out = jacobian(lambda xx: self.f(xx, u), x)
        self.jacobian_state_cache = out
        return self.jacobian_state_cache

    def jacobian_action(self, x, u, w=None):
        if self.num_action == 0:
            return np.zeros((self.num_constraint, 0))
        u = np.asarray(u, dtype=float)
        if self.num_parameter > 0:
            out = jacobian(lambda uu: self.f(x, uu, w), u)
        else:
            out = jacobian(lambda uu: self.f(x, uu), u)
        self.jacobian_action_cache = out
        return self.jacobian_action_cache


def constraint_eval(violations, constraints, states, actions, parameters=None):
    for t, con in enumerate(constraints):
        if con.num_constraint == 0:
            continue

        if parameters is None:
            con.evaluate_cache = con.evaluate(states[t], actions[t])
        else:
            con.evaluate_cache = con.evaluate(states[t], actions[t], parameters[t])

        violations[t] = con.evaluate_cache
    return violations


def jacobian_const(jx, ju, constraints, states, actions, parameters=None):
    H = len(constraints)

    for t, con in enumerate(constraints):
        if con.num_constraint == 0:
            continue

        if parameters is None:
            con.jacobian_state_cache = con.jacobian_state(states[t], actions[t])
            jx[t] = con.jacobian_state_cache

            if t < H - 1:
                con.jacobian_action_cache = con.jacobian_action(states[t], actions[t])
                ju[t] = con.jacobian_action_cache
        else:
            con.jacobian_state_cache = con.jacobian_state(
                states[t], actions[t], parameters[t]
            )
            jx[t] = con.jacobian_state_cache

            if t < H - 1:
                con.jacobian_action_cache = con.jacobian_action(
                    states[t], actions[t], parameters[t]
                )
                ju[t] = con.jacobian_action_cache
    return jx, ju
