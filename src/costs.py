import numpy as np

from .finite_diff import gradient, hessian, jacobian


class Cost:
    def __init__(self, f, num_state, num_action, num_parameter=0):
        """
        f: cost function f(x, u, [w]) -> scalar
        num_state: dimension of state
        num_action: dimension of action
        num_parameter: optional parameter dimension
        """
        self.f = f
        self.num_state = num_state
        self.num_action = num_action
        self.num_parameter = num_parameter

        self.evaluate_cache = np.zeros((1,))
        self.gradient_state_cache = np.zeros((num_state,))
        self.gradient_action_cache = np.zeros((num_action,))
        self.hessian_state_state_cache = np.zeros((num_state, num_state))
        self.hessian_action_action_cache = np.zeros((num_action, num_action))
        self.hessian_action_state_cache = np.zeros((num_action, num_state))

    def evaluate(self, x, u, w=None):
        val = self.f(x, u, w) if self.num_parameter > 0 else self.f(x, u)
        return float(np.asarray(val))

    def gradient_state(self, x, u, w=None):
        if self.num_state == 0:
            return np.zeros((0,))
        x = np.asarray(x, dtype=float)
        if self.num_parameter > 0:
            return gradient(lambda xx: self.f(xx, u, w), x)
        return gradient(lambda xx: self.f(xx, u), x)

    def gradient_action(self, x, u, w=None):
        if self.num_action == 0:
            return np.zeros((0,))
        u = np.asarray(u, dtype=float)
        if self.num_parameter > 0:
            return gradient(lambda uu: self.f(x, uu, w), u)
        return gradient(lambda uu: self.f(x, uu), u)

    def hessian_state_state(self, x, u, w=None):
        if self.num_state == 0:
            return np.zeros((0, 0))
        x = np.asarray(x, dtype=float)
        if self.num_parameter > 0:
            return hessian(lambda xx: self.f(xx, u, w), x)
        return hessian(lambda xx: self.f(xx, u), x)

    def hessian_action_action(self, x, u, w=None):
        if self.num_action == 0:
            return np.zeros((0, 0))
        u = np.asarray(u, dtype=float)
        if self.num_parameter > 0:
            return hessian(lambda uu: self.f(x, uu, w), u)
        return hessian(lambda uu: self.f(x, uu), u)

    def hessian_action_state(self, x, u, w=None):
        if self.num_action == 0 or self.num_state == 0:
            return np.zeros((self.num_action, self.num_state))
        x = np.asarray(x, dtype=float)
        if self.num_parameter > 0:
            return jacobian(lambda xx: self.gradient_action(xx, u, w), x)
        return jacobian(lambda xx: self.gradient_action(xx, u), x)


def cost(costs, states, actions, parameters=None):
    J = 0.0
    for t, c in enumerate(costs):
        if parameters is None:
            c.evaluate_cache = c.evaluate(states[t], actions[t])
        else:
            c.evaluate_cache = c.evaluate(states[t], actions[t], parameters[t])
        J += (
            float(c.evaluate_cache[0])
            if np.asarray(c.evaluate_cache).ndim > 0
            else float(c.evaluate_cache)
        )
    return J


def cost_gradient(gx, gu, costs, states, actions, parameters):
    H = len(costs)
    for t, c in enumerate(costs):
        if parameters is None:
            c.gradient_state_cache = c.gradient_state(states[t], actions[t])
            gx[t] = c.gradient_state_cache
            if t < H - 1:
                c.gradient_action_cache = c.gradient_action(states[t], actions[t])
                gu[t] = c.gradient_action_cache
        else:
            c.gradient_state_cache = c.gradient_state(
                states[t], actions[t], parameters[t]
            )
            gx[t] = c.gradient_state_cache
            if t < H - 1:
                c.gradient_action_cache = c.gradient_action(
                    states[t], actions[t], parameters[t]
                )
                gu[t] = c.gradient_action_cache


def cost_hessian(gxx, guu, gux, costs, states, actions, parameters):
    H = len(costs)
    for t, c in enumerate(costs):
        if parameters is None:
            # state-state Hessian
            c.hessian_state_state_cache = c.hessian_state_state(states[t], actions[t])
            gxx[t] = c.hessian_state_state_cache
            if t < H - 1:
                # action-action Hessian
                c.hessian_action_action_cache = c.hessian_action_action(
                    states[t], actions[t]
                )
                guu[t] = c.hessian_action_action_cache

                # action-state Hessian
                c.hessian_action_state_cache = c.hessian_action_state(
                    states[t], actions[t]
                )
                gux[t] = c.hessian_action_state_cache

        else:
            # state-state Hessian
            c.hessian_state_state_cache = c.hessian_state_state(
                states[t], actions[t], parameters[t]
            )
            gxx[t] += c.hessian_state_state_cache
            if t < H - 1:
                # action-action Hessian
                c.hessian_action_action_cache = c.hessian_action_action(
                    states[t], actions[t], parameters[t]
                )
                guu[t] = c.hessian_action_action_cache

                # action-state Hessian
                c.hessian_action_state_cache = c.hessian_action_state(
                    states[t], actions[t], parameters[t]
                )
                gux[t] = c.hessian_action_state_cache
