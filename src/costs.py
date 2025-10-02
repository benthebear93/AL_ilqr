import jax
import jax.numpy as jnp


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

        # gradient
        self.grad_state = jax.grad(f, argnums=0)
        self.grad_action = jax.grad(f, argnums=1)

        # hessian
        self.hess_state_state = jax.hessian(f, argnums=0)
        self.hess_action_action = jax.hessian(f, argnums=1)

        self.evaluate_cache = jnp.zeros((1,))
        self.gradient_state_cache = jnp.zeros((num_state,))
        self.gradient_action_cache = jnp.zeros((num_action,))
        self.hessian_state_state_cache = jnp.zeros((num_state, num_state))
        self.hessian_action_action_cache = jnp.zeros((num_action, num_action))
        self.hessian_action_state_cache = jnp.zeros((num_action, num_state))

        def grad_action_fn(x, u, w=None):
            return (
                jax.grad(f, argnums=1)(x, u, w)
                if num_parameter > 0
                else jax.grad(f, argnums=1)(x, u)
            )

        self.hess_action_state = jax.jacfwd(grad_action_fn, argnums=0)

    def evaluate(self, x, u, w=None):
        return self.f(x, u, w) if self.num_parameter > 0 else self.f(x, u)

    def gradient_state(self, x, u, w=None):
        return (
            self.grad_state(x, u, w)
            if self.num_parameter > 0
            else self.grad_state(x, u)
        )

    def gradient_action(self, x, u, w=None):
        return (
            self.grad_action(x, u, w)
            if self.num_parameter > 0
            else self.grad_action(x, u)
        )

    def hessian_state_state(self, x, u, w=None):
        return (
            self.hess_state_state(x, u, w)
            if self.num_parameter > 0
            else self.hess_state_state(x, u)
        )

    def hessian_action_action(self, x, u, w=None):
        return (
            self.hess_action_action(x, u, w)
            if self.num_parameter > 0
            else self.hess_action_action(x, u)
        )

    def hessian_action_state(self, x, u, w=None):
        return (
            self.hess_action_state(x, u, w)
            if self.num_parameter > 0
            else self.hess_action_state(x, u)
        )


def cost(costs, states, actions, parameters=None):
    J = 0.0
    for t, c in enumerate(costs):
        if parameters is None:
            c.evaluate_cache = c.evaluate(states[t], actions[t])
        else:
            c.evaluate_cache = c.evaluate(states[t], actions[t], parameters[t])
        J += (
            float(c.evaluate_cache[0])
            if c.evaluate_cache.ndim > 0
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
