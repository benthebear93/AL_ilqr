import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())

x = jnp.ones((1000, 1000))
y = jnp.dot(x, x.T)
print("Dot product result:", y.shape)
