import jax
import jax.numpy as jnp


class Dynamics:
    def __init__(self, f, num_state, num_action, num_parameter=0, num_next_state=None):
        self.f = f
        self.num_state = num_state
        self.num_action = num_action
        self.num_parameter = num_parameter
        self.num_next_state = num_state if num_next_state is None else num_next_state

        # define jacobians
        self._jac_state = jax.jacfwd(f, argnums=0)
        self._jac_action = jax.jacfwd(f, argnums=1)
        if num_parameter > 0:
            self._jac_param = jax.jacfwd(f, argnums=2)
        else:
            self._jac_param = None

        self.evaluate_cache = jnp.zeros((self.num_next_state,))
        self.jacobian_state_cache = jnp.zeros((self.num_next_state, self.num_state))
        self.jacobian_action_cache = jnp.zeros((self.num_next_state, self.num_action))

    def evaluate(self, x, u, w=None):
        if self.num_parameter > 0:
            out = self.f(x, u, w)
        else:
            out = self.f(x, u)
        self.evaluate_cache = out
        return self.evaluate_cache

    def jacobian_state(self, x, u, w=None):
        if self.num_parameter > 0:
            out = self._jac_state(x, u, w)
        else:
            out = self._jac_state(x, u)
        self.jacobian_state_cache = out
        return self.jacobian_state_cache

    def jacobian_action(self, x, u, w=None):
        if self.num_parameter > 0:
            out = self._jac_action(x, u, w)
        else:
            out = self._jac_action(x, u)
        self.jacobian_action_cache = out
        return self.jacobian_action_cache

    def jacobian_parameter(self, x, u, w):
        if self._jac_param is None:
            raise ValueError("This dynamics has no parameters")
        return self._jac_param(x, u, w)


def dynamics_eval(d: Dynamics, x, u, w=None):
    return d.evaluate(x, u, w)


def jacobian_model(jx, ju, dynamics, x, u, w):
    for t, d in enumerate(dynamics):
        if d.num_parameter > 0:
            d.jacobian_state_cache = d.jacobian_state(x[t], u[t], w[t])
            d.jacobian_action_cache = d.jacobian_action(x[t], u[t], w[t])
        else:
            d.jacobian_state_cache = d.jacobian_state(x[t], u[t])
            d.jacobian_action_cache = d.jacobian_action(x[t], u[t])

        # Julia: @views jacobian_states[t] .= d.jacobian_state_cache
        jx[t] = d.jacobian_state_cache
        ju[t] = d.jacobian_action_cache


def num_trajectory(dynamics):
    return (
        sum(d.num_state + d.num_action for d in dynamics) + dynamics[-1].num_next_state
    )


def init_dynamics(dynamics, num_state, num_action):
    return Dynamics(dynamics, num_state, num_action)


class DynamicsUserDefined:
    def __init__(
        self, f, fx, fu, num_next_state, num_state, num_action, num_parameter=0
    ):
        self.f = f
        self._jac_state = fx  # user-provided function
        self._jac_action = fu  # user-provided function
        self.num_next_state = num_next_state
        self.num_state = num_state
        self.num_action = num_action
        self.num_parameter = num_parameter

        self.evaluate_cache = jnp.zeros((self.num_next_state,))
        self.jacobian_state_cache = jnp.zeros((self.num_next_state, self.num_state))
        self.jacobian_action_cache = jnp.zeros((self.num_next_state, self.num_action))

    def evaluate(self, x, u, w=None):
        if self.num_parameter > 0:
            out = self.f(x, u, w)
        else:
            out = self.f(x, u)
        self.evaluate_cache = out
        return self.evaluate_cache

    def jacobian_state(self, x, u, w=None):
        if self.num_parameter > 0:
            out = self._jac_state(x, u, w)
        else:
            out = self._jac_state(x, u)
        self.jacobian_state_cache = out
        return self.jacobian_state_cache

    def jacobian_action(self, x, u, w=None):
        if self.num_parameter > 0:
            out = self._jac_action(x, u, w)
        else:
            out = self._jac_action(x, u)
        self.jacobian_action_cache = out
        return self.jacobian_action_cache


def init_user_defined_dynamics(
    f, fx, fu, num_next_state, num_state, num_action, num_parameter=0
):
    """
    Initialize dynamics with user-provided function and its Jacobians.
    f  : state transition function
    fx : Jacobian wrt state
    fu : Jacobian wrt action
    """
    return DynamicsUserDefined(
        f, fx, fu, num_next_state, num_state, num_action, num_parameter
    )
