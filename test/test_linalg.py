
# Quu[t]  [[ 0.0746408  -0.00019741]
#  [ 0.00019742  0.07464081]]
# Qux[t] [[ 0.48134732  0.2585864  -0.00197404]
#  [ 0.00100668  0.00230219  0.5464082 ]]
# L [[2.7320465e-01 0.0000000e+00]
#  [2.2264262e-08 2.7320468e-01]]
# y [[ 1.7618562   0.9464934  -0.0072255 ]
#  [ 0.00368458  0.00842655  1.9999956 ]]
# K[t] [[-6.448851   -3.4644117   0.02644782]
#  [-0.01348653 -0.03084334 -7.3205023 ]]
# import numpy as np
# import jax.numpy as jnp
# from jax.scipy.linalg import solve_triangular

# Quu_t = jnp.array([
#     [0.0746408, -0.00019741],
#     [0.00019742, 0.07464081]
# ])

# Qux_t = jnp.array([
#     [0.48134732, 0.2585864, -0.00197404],
#     [0.00100668, 0.00230219, 0.5464082]
# ])

# # Cholesky 하삼각 (JAX는 L 반환)
# L = jnp.linalg.cholesky(Quu_t)

# # Solve (Julia potrs!와 동일 동작)
# y = solve_triangular(L, Qux_t, lower=True)
# K = -solve_triangular(L.T, y, lower=False)

# print("L =\n", np.array(L))
# print("K =\n", np.array(K))

import numpy as np
from scipy.linalg.lapack import dpotrf, dpotrs
np.set_printoptions(precision=12, suppress=True)
# Quu_t = np.array([
#     [0.0746408, -0.00019741],
#     [0.00019742, 0.07464081]
# ])
Quu_t = np.array(
    [[20.03,     0.      ],
 [ 0.       ,20.030006]]
)
# Qux_t = np.array([
#     [0.48134732, 0.2585864, -0.00197404],
#     [0.00100668, 0.00230219, 0.5464082]
# ])

Qux_t = np.array(
    [[200.09755 ,       0.99049103,     0.          ],
 [ -0.0004952455,   0.10004877 ,  200.10011     ]]
)
# Julia: potrf!('U') → 상삼각
U, info = dpotrf(Quu_t, lower=0)
print("U =\n", U.flatten())

# Julia: potrs!('U', U, Qux)
K, info = dpotrs(U, Qux_t, lower=0)
K *= 1.0
print("K =\n", K.flatten())
